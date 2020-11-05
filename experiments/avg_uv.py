# -----------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Xueting Li.
# -----------------------------------------------------------

# Script to compute a semantic template given trained reconstruction network.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags

from ..nnutils import loss_utils
from ..nnutils import test_utils
from ..nnutils import geom_utils
from ..nnutils.smr import SoftRenderer
from ..nnutils import cub_mesh_s1 as mesh_net

from ..data import cub as cub_data
from ..utils import tf_visualizer
from ..utils import transformations
from ..utils import image as image_utils

import os
import cv2
import random
import scipy.misc
import torchvision
import numpy as np
from tqdm import tqdm
import os.path as osp
from PIL import Image
from collections import OrderedDict

import soft_renderer as sr
import neural_renderer as nr
import soft_renderer.functional as srf

import torch
import torch.nn as nn
import torchvision.utils as vutils

flags.DEFINE_integer('image_size', 256, 'training image size')
flags.DEFINE_boolean('pred_cam', True, 'If true predicts camera')
flags.DEFINE_integer('axis', 1, 'symmetric axis')
flags.DEFINE_integer('num_parts', 4, 'number of semantic parts')
flags.DEFINE_boolean('use_scops', True, 'If true load SCOPS in loader')
flags.DEFINE_string('model_path', 'unsup-mesh/cachedir/snapshots/cub_net/pred_net_latest.pth', 'model path')
flags.DEFINE_string('out_dir', 'unsup-mesh/cachedir/snapshots/cub_net/', 'output directory')

opts = flags.FLAGS

class ShapenetTester(test_utils.Tester):
    def define_model(self):
        """
        Define the model

        Args:
            self: (todo): write your description
        """
        opts = self.opts

        # define model
        img_size = (opts.image_size, opts.image_size)
        self.model = mesh_net.MeshNet(
            img_size, opts, nz_feat=opts.nz_feat,
            axis = opts.axis)

        self.load_my_state_dict(opts.model_path)
        self.model = self.model.cuda(device=opts.gpu_id)
        self.mean_shape = self.model.get_mean_shape()
        self.faces = self.model.faces.view(1, -1, 3)

        # define differentiable renderer
        self.renderer = SoftRenderer(opts.image_size, 'softmax')

        # define colorization tools
        self.colorize = image_utils.Colorize(opts.num_parts + 1)
        self.batch_colorize = image_utils.BatchColorize(opts.num_parts + 1)

        # define criterion functions
        self.texture_loss_fn = loss_utils.PerceptualTextureLoss()
        os.makedirs(opts.out_dir, exist_ok=True)

        return

    def init_dataset(self):
        """
        Initialize the module.

        Args:
            self: (todo): write your description
        """
        opts = self.opts
        self.data_module = cub_data
        self.dataloader = self.data_module.data_loader(opts)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def load_my_state_dict(self, resume_dir):
        """
        Loads a dictionary of the states_dir.

        Args:
            self: (todo): write your description
            resume_dir: (str): write your description
        """
        saved_state_dict = torch.load(resume_dir)
        # registered buff tensors may have different batch size in training
        # so we do not load them from the pretrained models
        unwanted_keys = {"noise", "uv_sampler"}

        new_params = self.model.state_dict().copy()
        for name, param in new_params.items():
            if name not in unwanted_keys:
                new_params[name].copy_(saved_state_dict[name])
        self.model.load_state_dict(new_params)
        print(tf_visualizer.green("Loaded model from {}.".format(resume_dir)))

    def set_input(self, batch):
        """
        Set the input tensor.

        Args:
            self: (todo): write your description
            batch: (todo): write your description
        """
        opts = self.opts

        input_img_tensor = batch['img'].type(torch.FloatTensor)
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])
        img_tensor = batch['img'].type(torch.FloatTensor)
        mask_tensor = batch['mask'].type(torch.FloatTensor)
        part_prob_tensor = batch['part_prob'].type(torch.FloatTensor)

        self.input_imgs = input_img_tensor.cuda()
        self.imgs = img_tensor.cuda()
        self.masks = mask_tensor.cuda()
        img_path = batch['img_path']
        self.part_segs = part_prob_tensor.permute(0, 3, 1, 2)

    def most_freq(self, arr):
        """
        Given an arr of N * D, return a N dimension array,
        indicating the most frequent element in 2nd dimension.
        """
        n,d = arr.size()
        k = torch.max(torch.unique(arr)) + 1
        arr_one_hot = torch.zeros(n * d, k).cuda()
        arr_flat = arr.view(-1, 1)
        arr_one_hot.scatter_(1, arr_flat, 1)
        arr_one_hot = arr_one_hot.view(n, d, k)
        arr_ = torch.sum(arr_one_hot, dim = 1)
        return torch.argmax(arr_, dim = 1)

    def compute_uv(self):
        """
        Compute the model to predict.

        Args:
            self: (todo): write your description
        """
        dataloader = iter(self.dataloader)

        self.best_shape = None
        best_idx = 0
        best_mask_loss = 100000.0
        self.best_uv = None

        print(tf_visualizer.green("Search for the examplar instance."))
        for i, batch in tqdm(enumerate(self.dataloader)):
            self.set_input(batch)

            with torch.no_grad():
                outputs = self.model(self.input_imgs)
                # shape
                delta_v = outputs['delta_v']

                if(opts.symmetric):
                    delta_v = self.model.symmetrize(delta_v)
                    self.mean_shape = self.model.get_mean_shape()

                pred_vs = self.mean_shape + delta_v

                # camera
                proj_cam = outputs['cam']

                faces = self.faces.repeat(delta_v.size(0), 1, 1)
                ori_flip = torch.FloatTensor([1, -1, 1, 1, 1, -1, -1]).view(1,-1).cuda()
                proj_cam = proj_cam * ori_flip
                pred_seen, _, _ = self.renderer.forward(pred_vs, faces, proj_cam)
                rgb_pred_seen = pred_seen[:, 0:3, :, :]
                mask_pred_seen = pred_seen[:, 3, :, :]

                # flip images
                flip_imgs = torch.flip(self.imgs, [3])
                flip_masks = torch.flip(self.masks, [2])
                texture_loss = self.texture_loss_fn(rgb_pred_seen, flip_imgs, flip_masks, mask_pred_seen, avg = False)

                # the new best shape should also be close to the old one,
                # because we don't want the template to change all the time.
                mean_shape = self.mean_shape.unsqueeze(0).repeat(pred_vs.size(0), 1, 1)
                dist = torch.nn.MSELoss(reduction='none')(pred_vs, mean_shape)
                # scale is used to make sure these two losses are comparable
                scale = texture_loss.mean() / torch.sum(dist, dim = (1, 2)).mean()
                texture_loss += (torch.sum(dist, dim = (1, 2))) * scale

                # select a shape that has both low mask loss and close to the current best_shape
                min_mask, min_idx = torch.min(texture_loss, 0)
                if(best_mask_loss > min_mask):
                    best_idx = min_idx
                    best_mask_loss = min_mask
                    self.best_shape = pred_vs[best_idx].unsqueeze(0)
                    uvimage_pred = outputs['uvimage_pred']
                    uv_parts = torch.nn.functional.grid_sample(self.part_segs.cuda(), uvimage_pred.permute(0, 2, 3, 1))
                    self.best_uv = uv_parts[best_idx].unsqueeze(0)

                    # visulize semantic texture
                    tex_flow = outputs['tex_flow'][best_idx].unsqueeze(0)
                    tex = geom_utils.sample_textures(tex_flow, self.part_segs[best_idx].unsqueeze(0).cuda())
                    best_tex = tex.contiguous()

                    bs, fs, ts, _, cs = best_tex.size()
                    best_tex = best_tex.view(bs, fs, -1, cs)
                    best_tex = torch.argmax(best_tex, dim = -1)
                    self.best_tex = self.batch_colorize(best_tex.cpu())
                    self.best_tex = self.best_tex.permute(0, 2, 3, 1)

        self.best_shape = self.best_shape.repeat(opts.batch_size, 1, 1)
        self.best_uv = self.best_uv.repeat(opts.batch_size, 1, 1, 1)

        print(tf_visualizer.green("Start to compute semantic template."))
        counter = 0
        avg_uv_parts = None
        for i, batch in tqdm(enumerate(self.dataloader)):
            self.set_input(batch)

            with torch.no_grad():
                outputs = self.model(self.input_imgs)

                self.uvimage_pred = outputs['uvimage_pred']
                uv_parts = torch.nn.functional.grid_sample(self.part_segs.cuda(), self.uvimage_pred.permute(0, 2, 3, 1))

                uv_parts_ch = uv_parts.clone()
                best_uv_ch = self.best_uv.clone()
                dist = torch.nn.MSELoss(reduction='none')(uv_parts_ch, best_uv_ch)
                dist = torch.sum(dist, dim = (1, 2, 3))
                _, idx = torch.topk(dist, k = 5, largest = False)

                if(avg_uv_parts is None):
                    avg_uv_parts = torch.sum(uv_parts[idx, :, :, :], dim = 0).unsqueeze(0)
                else:
                    avg_uv_parts += torch.sum(uv_parts[idx, :, :, :], dim = 0).unsqueeze(0)
                counter += idx.size(0)

        avg_prob = avg_uv_parts / counter

        avg_prob = avg_prob.cpu().squeeze().numpy()
        avg_prob = avg_prob.transpose(1,2,0)
        uv_path = osp.join(opts.out_dir, "semantic_prob.npy")
        np.save(uv_path, avg_prob)
        avg_prob = np.asarray(np.argmax(avg_prob, axis=2), dtype=np.int)

        pil_image = Image.fromarray(avg_prob.astype(dtype=np.uint8))
        pil_image.save(osp.join(opts.out_dir, "semantic_seg.png"), 'PNG')

        color_vis = self.colorize(avg_prob)
        color_vis = torch.from_numpy(color_vis).float()

        # wrap the uv map onto template
        uv_sampler = self.model.uv_sampler[0].unsqueeze(0)
        tex = torch.nn.functional.grid_sample(color_vis.unsqueeze(0).cuda().float(), uv_sampler)
        tex = tex.view(tex.size(0), -1, tex.size(2), opts.tex_size, opts.tex_size).permute(0, 2, 3, 4, 1)
        tex_left = tex[:, -self.model.texture_predictor.num_sym_faces:]
        tex = torch.cat([tex, tex_left], 1)
        tex = tex.view(tex.size(0), tex.size(1), -1, 3)

        mean_v = self.model.get_mean_shape()
        mesh_ = sr.Mesh(mean_v, self.faces, tex)
        mesh_path = osp.join(opts.out_dir,"mean_template.obj")
        mesh_.save_obj(mesh_path, save_texture=True)

        # compute vertices/faces belong to each part
        uv_label = np.load(uv_path)
        uv_label = torch.from_numpy(uv_label).float().unsqueeze(0).permute(0, 3, 1, 2)
        uv_label = uv_label.cuda()
        tex_seg = torch.nn.functional.grid_sample(uv_label, uv_sampler)
        tex_seg = tex_seg.view(tex_seg.size(0), -1, tex_seg.size(2), opts.tex_size, opts.tex_size).permute(0, 2, 3, 4, 1)
        tex_left = tex_seg[:, -self.model.texture_predictor.num_sym_faces:]
        tex_seg = torch.cat([tex_seg, tex_left], 1)
        tex_seg = tex_seg.view(tex_seg.size(0), tex_seg.size(1), -1, (self.opts.num_parts + 1))
        tex_seg = torch.argmax(tex_seg, dim = -1)

        # obtain vertex label through face label
        tex_seg = self.most_freq(tex_seg.squeeze())
        tex_seg = tex_seg.float()
        face = self.faces[0]
        parts = []
        for cnt in range(opts.num_parts):
            parts.append([])

        # go through all vertices and compute their label
        # for sanity check
        vert_tex = [] 
        for cnt in range(face.max() + 1):
            v0 = (face[:, 0] == cnt) * 1
            v1 = (face[:, 1] == cnt) * 1
            v2 = (face[:, 2] == cnt) * 1
            v = v0 + v1 + v2
            # which faces relate to this vertex
            idxes = torch.nonzero(v).squeeze()
            labels = tex_seg[idxes].long().view(1, idxes.size(0))
            label = self.most_freq(labels)

            if(label > 0):
                parts[label-1].append(cnt)
            vert_tex.append(label)

        np.save(osp.join(opts.out_dir, "head_vertices.npy"), parts[0])
        np.save(osp.join(opts.out_dir, "neck_vertices.npy"), parts[1])
        np.save(osp.join(opts.out_dir, "back_vertices.npy"), parts[2])
        np.save(osp.join(opts.out_dir, "belly_vertices.npy"), parts[3])

        # visualize part label for each vertex
        vert_tex = torch.stack(vert_tex)
        vert_tex = self.colorize(vert_tex.view(642, 1).cpu().numpy())
        vert_tex = torch.from_numpy(vert_tex).float().squeeze()
        vert_tex = vert_tex.permute(1, 0)

        mesh_ = sr.Mesh(mean_v, self.faces, vert_tex.view(1, 642, 3), texture_type='vertex')
        mesh_path = osp.join(opts.out_dir,"vertex_label.obj")
        mesh_.save_obj(mesh_path, save_texture=True)

        print(tf_visualizer.green("Semantic template saved at {}.".format(opts.out_dir)))

def set_seed(seed):
    """
    Benchmark a random seed.

    Args:
        seed: (int): write your description
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(_):
    """
    Main function.

    Args:
        _: (int): write your description
    """
    set_seed(0)
    tester = ShapenetTester(opts)
    tester.init_testing()
    tester.compute_uv()

if __name__ == '__main__':
    app.run(main)
