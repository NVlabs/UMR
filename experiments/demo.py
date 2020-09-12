# ------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Xueting Li.
# ------------------------------------------------------------

# Demo Script
# Inputs:
#  - single view images
# Outputs:
#  - Reconstructed shape
#  - Reconstructed camera pose
#  - Reconstructed texture
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import cv2
import math
import torch
import random
import imageio
import torchvision
import numpy as np
import os.path as osp
from PIL import Image
from absl import app, flags
import torchvision.utils as vutils
from collections import OrderedDict

from ..nnutils import test_utils
from ..nnutils import geom_utils
from ..nnutils import cub_mesh as mesh_net
from ..nnutils.nmr_pytorch import NeuralRenderer

from ..utils import tf_visualizer
from ..utils import image as image_utils

flags.DEFINE_integer('axis', 1, 'symmetric axis')
flags.DEFINE_integer('image_size', 256, 'training image size')
flags.DEFINE_boolean('use_scops', False, 'If true load SCOPS')
flags.DEFINE_boolean('pred_cam', True, 'If true predicts camera')
flags.DEFINE_string('img_path', 'unsup-mesh/demo_imgs/bird.png', 'input image')
flags.DEFINE_string('out_path', 'unsup-mesh/cachedir/demo/', 'output path')
flags.DEFINE_string('model_path', 'unsup-mesh/cachedir/snapshots/cub_net/pred_net_latest.pth', 'model path')

opts = flags.FLAGS

class ShapenetTester(test_utils.Tester):
    def define_model(self):
        opts = self.opts

        # define model
        img_size = (opts.image_size, opts.image_size)
        self.model = mesh_net.MeshNet(img_size, opts, nz_feat=opts.nz_feat, axis = opts.axis)
        self.model.eval()
        self.load_my_state_dict(os.path.join(opts.model_path))
        self.mean_shape = self.model.get_mean_shape()
        self.model = self.model.cuda(device=opts.gpu_id)

        # define renderer
        if opts.use_texture:
            self.tex_renderer = NeuralRenderer(opts.image_size)
            self.tex_renderer.ambient_light_only()
            self.tex_renderer.set_bgcolor([1, 1, 1])
            self.tex_renderer.set_light_dir([0, 1, -1], 0.4)

        # default texture and camera
        t_size = opts.tex_size
        default_tex = np.ones((1, 1280, t_size, t_size, 3))
        blue = np.array([156, 199, 234.]) / 255.
        default_tex = torch.from_numpy(default_tex * blue).float().cuda()
        self.default_tex = default_tex.view(1, 1280, t_size, t_size, 3).unsqueeze(4).repeat(opts.batch_size, 1, 1, 1, t_size, 1)

        angles = torch.Tensor([math.radians(60), 0, 0]).view(1,3)
        self.base_rotation = self.ang2quat(angles).view(1, 4).repeat(opts.batch_size, 1)

        return

    def load_my_state_dict(self, resume_dir):
        saved_state_dict = torch.load(resume_dir)
        # the tensors may have different batch size
        unwanted_keys = {"uv_sampler"}

        new_params = self.model.state_dict().copy()
        for name, param in new_params.items():
            if name not in unwanted_keys:
                new_params[name].copy_(saved_state_dict[name])
        self.model.load_state_dict(new_params)
        print(tf_visualizer.green("Loaded model from {}.".format(resume_dir)))

    def init_dataset(self):
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def demo(self):
        # load image
        img = self.preprocess_image()
        batch = {'img': torch.Tensor(np.expand_dims(img, 0)).repeat(opts.batch_size, 1, 1, 1)}
        input_img_tensor = batch['img'].type(torch.FloatTensor).clone()
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])
        img_tensor = batch['img'].type(torch.FloatTensor)
        self.input_imgs = input_img_tensor.cuda(device=opts.gpu_id)
        self.imgs = img_tensor.cuda(device=opts.gpu_id)

        os.makedirs(self.opts.out_path, exist_ok = True)
        with torch.no_grad():
            # forward
            outputs = self.model.forward(self.input_imgs)
            # camera
            self.cams = outputs['cam']

            # texture
            if(self.opts.use_texture):
                self.tex_flow = outputs['tex_flow']
                self.tex = geom_utils.sample_textures(self.tex_flow, self.imgs)
                self.tex = self.tex.contiguous()

            bs, fs, ts, _, _ = self.tex.size()
            self.tex = self.tex.view(bs, fs, -1, 3)

            # shape
            self.delta_v = outputs['delta_v']
            if(opts.symmetric):
                delta_v = self.model.symmetrize(self.delta_v)
            self.verts = self.mean_shape + delta_v
            faces = self.model.faces.view(1, -1, 3)
            self.faces = faces.repeat(opts.batch_size, 1, 1)

            # render using predicted camera and default texture
            pred_tex = self.imgs[0][:, :opts.image_size, :opts.image_size]
            texture_rgba = self.tex_renderer.forward(self.verts, self.faces, self.cams, self.default_tex)
            pred_ = texture_rgba[:, 0:3, :, :][0]
            pred_tex = torch.cat((pred_tex, pred_), dim = 2)

            # render using predicted camera and predicted texture
            nb, nf, _, nc = self.tex.size()
            tex = self.tex.view(nb, nf, opts.tex_size, opts.tex_size, nc).unsqueeze(4).repeat(1, 1, 1, 1, 6, 1)
            texture_rgba = self.tex_renderer.forward(self.verts, self.faces, self.cams, tex)
            pred_ = texture_rgba[:, 0:3, :, :][0]
            pred_tex = torch.cat((pred_tex, pred_), dim = 2)

            # render under unobserved views with predicted texture
            cam_biases = [self.cams[:, 3:].unsqueeze(1)]
            for i in range(1,4):
                cam_biases.append(geom_utils.hamilton_product(self.base_rotation.unsqueeze(1).cuda(), cam_biases[i-1]))

            cam_biases = torch.cat(cam_biases, dim = 1)
            self.other_cams = torch.cat((self.cams[:, :3].unsqueeze(1).repeat(1, 4, 1), cam_biases), dim = -1).cuda()

            # concatenate everything and save
            pred_tex_ = None
            for cnt in range(1, self.other_cams.size(1)):
                texture_rgba = self.tex_renderer.forward(self.verts, self.faces, self.other_cams[:, cnt, :], tex)
                pred_ = texture_rgba[:, 0:3, :, :][0]
                if(cnt == 1):
                    pred_tex_ = pred_
                else:
                    pred_tex_ = torch.cat((pred_tex_, pred_), dim = 2)

            pred_tex = torch.cat((pred_tex, pred_tex_), dim = 1)

            self.save_image(pred_tex, os.path.join(self.opts.out_path, "pred_tex.png"))
            print(tf_visualizer.green("Results saved at {}.".format(self.opts.out_path)))

    def save_image(self, img, path):
        vutils.save_image(img, path)

    def ang2quat(self, angles):
        # convert from Eular angles to quaternion
        axis = torch.eye(3).float().cuda()
        q_az = geom_utils.convert_ax_angle_to_quat(axis[1], angles[...,0])
        q_el = geom_utils.convert_ax_angle_to_quat(axis[0], angles[...,1])
        q_cr = geom_utils.convert_ax_angle_to_quat(axis[2], angles[...,2])
        quat = geom_utils.hamilton_product(q_el.unsqueeze(1), q_az.unsqueeze(1))
        quat = geom_utils.hamilton_product(q_cr.unsqueeze(1), quat)
        quat = quat.squeeze(1)
        return quat

    def preprocess_image(self):
        # load image from disk, we assume object is centered
        print(tf_visualizer.green("Read image from {}.".format(self.opts.img_path)))
        img = imageio.imread(self.opts.img_path) / 255.0
        img_size = self.opts.image_size

        # Scale the max image size to be img_size
        scale_factor = float(img_size) / np.max(img.shape[:2])
        img, _ = image_utils.resize_img(img, scale_factor)

        # Crop img_size x img_size from the center
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # img center in (x, y)
        center = center[::-1]
        bbox = np.hstack([center - img_size / 2., center + img_size / 2.])

        img = image_utils.crop(img, bbox, bgval=1.)

        # Transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))

        return img

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(_):
    set_seed(0)
    tester = ShapenetTester(opts)
    tester.init_testing()
    tester.demo()

if __name__ == '__main__':
    app.run(main)
