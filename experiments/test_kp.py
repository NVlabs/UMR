# ------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Xueting Li.
# ------------------------------------------------------------

# Testing Script for quantitative evaluations
# Inputs:
#  - single view images
# Evaluation metric:
#  - keypoint transfer PCK using predicted camera pose
#  - keypoint transfer PCK using predicted texture flow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags

from ..nnutils import test_utils
from ..nnutils import geom_utils
from ..nnutils import chamfer_python
from ..nnutils.smr import SoftRenderer
from ..nnutils import cub_mesh as mesh_net

from ..utils import kp_utils, tf_visualizer
from ..data import cub_kp_transfer as cub_data

import os
import cv2
import copy
import torch
import random
import torchvision
import numpy as np
import os.path as osp
from tqdm import tqdm
from collections import OrderedDict

flags.DEFINE_integer('image_size', 256, 'training image size')
flags.DEFINE_boolean('pred_cam', True, 'If true predicts camera')
flags.DEFINE_boolean('visualize', False, 'If true save keypoint visualization')
flags.DEFINE_integer('axis', 1, 'symmetric axis')
flags.DEFINE_integer('sigma', 3, 'sigma for converting key points to heat maps')
flags.DEFINE_string('mode', 'flow', 'flow|cam, use flow or cam to transfer keypoints.')
flags.DEFINE_boolean('use_scops', False, 'If true predicts camera')
flags.DEFINE_string('model_path', 'unsup-mesh/cachedir/snapshots/cub_net/pred_net_latest.pth', 'model path')


opts = flags.FLAGS

# For keypoint visualization
color_table = np.array([[0,255,0],
                        [0,0,255],
                        [0,255,255],
                        [255,0,0],
                        [255,0,255],
                        [255,255,0],
                        [0,0,255],
                        [0,128,255],
                        [128,0,255],
                        [0,128,0],
                        [128,0,0],
                        [0,0,128],
                        [128,128,0],
                        [0,128,128],
                        [128,0,128],
                        [128,128,128],
                        ])

def collate_pair_batch(examples):
    batch = {}
    for key in examples[0]:
        if key in ['kp_uv', 'img', 'inds', 'neg_inds', 'mask', 'kp', 'pos_inds', 'sfm_pose', 'anchor']:
            batch[key] = torch.cat([examples[i][key] for i in range(len(examples))], dim=0)
        elif key in ['img_path']:
            batch[key] = []
            for i in range(len(examples)):
                batch[key].append(examples[i][key])
    return batch

class ShapenetTester(test_utils.Tester):
    def define_model(self):
        opts = self.opts

        # define model
        self.symmetric = opts.symmetric
        img_size = (opts.image_size, opts.image_size)
        self.model = mesh_net.MeshNet(
            img_size, opts, nz_feat=opts.nz_feat,
            axis = opts.axis)
        self.load_my_state_dict(os.path.join(opts.model_path))
        self.model = self.model.cuda(device=opts.gpu_id)
        self.model.eval()
        self.mean_shape = self.model.get_mean_shape().unsqueeze(0)

        # define renderer
        self.renderer = SoftRenderer(opts.image_size, "softmax")

        return

    def load_my_state_dict(self, resume_dir):
        saved_state_dict = torch.load(resume_dir)
        # buffered tensors may have different batch size when inference
        # so we do not load them
        unwanted_keys = {"uv_sampler"}

        # only copy the params that exist in current model (caffe-like)
        new_params = self.model.state_dict().copy()
        for name, param in new_params.items():
            if name not in unwanted_keys:
                new_params[name].copy_(saved_state_dict[name])
        self.model.load_state_dict(new_params)
        print(tf_visualizer.green("Loaded model from {}.".format(resume_dir)))

    def init_dataset(self):
        opts = self.opts
        dataloader_fn = cub_data.cub_test_pair_dataloader
        self.dl_img1 = dataloader_fn(opts, 1)
        self.dl_img2 = dataloader_fn(opts, 2)

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def map_kp_img1_to_img2_flow(self, kp_src, flow_src, flow_tgt):
        """
        Mapping keypoints from image 1 to image 2 using flow.
        """
        # transfer keypoint from one image to another
        # map all points on target image to a face
        grid_size = torch.Size((1, 2, opts.image_size, opts.image_size))
        sgrid = kp_utils.create_grid(grid_size)
        sgrid = sgrid.permute(0, 3, 1, 2) # 32 x 2 x 256 x 256
        sgrid = sgrid.cuda()
        # sample standard grid to map each point onto faces
        nf = flow_tgt.size(0)
        p2face = torch.nn.functional.grid_sample(sgrid, flow_tgt.view(1, nf, -1, 2))
        # each face corresponds to a pixel on the image
        p2face = torch.mean(p2face, dim = -1).permute(0, 2, 1) # 2 x nf x 2
        p2face = p2face.cpu().squeeze()

        # map keypoints from source image onto faces
        # turn keypoints to a heat map
        kp_num = kp_src.size(0)
        hp = torch.zeros(1, kp_num, opts.image_size, opts.image_size)
        kp_src = (kp_src[:, 0:2] + 1) / 2.0 * 256
        for ccnt in range(kp_num):
            hp[0, ccnt] = kp_utils.draw_labelmap(hp[0, ccnt], (kp_src[ccnt][0], kp_src[ccnt][1]), sigma = opts.sigma)

        # sample from heat map
        k2face = torch.nn.functional.grid_sample(hp.cuda(), flow_src.view(1, nf, -1, 2))
        k2face = torch.mean(k2face, dim = -1).cpu()
        _, k2face_idx = torch.max(k2face, dim = -1)

        # finally we have keypoints->faces->target image
        k2k = p2face[k2face_idx]

        return k2k

    def map_kp_img1_to_img2_cam(self, kp_src, cam_src, cam_tgt, mask_tgt):
        """
        Map from image 1 to image 2 using camera & mean template
        """
        kp_num = kp_src.size(0)
        # project all vertices of the mean shape using target camera
        cam_src = cam_src.view(1, 7)
        cam_tgt = cam_tgt.view(1, 7)
        vert2ds_tgt = self.renderer.project_points(self.mean_shape, cam_tgt)

        # for each pixel in the target image foreground,
        # search its correspondence on all projected points
        grid_size = torch.Size((1, 2, opts.image_size, opts.image_size))
        sgrid2D = kp_utils.create_grid(grid_size).squeeze()
        sgrid = sgrid2D.view(-1, 2)

        mask_tgt = mask_tgt.view(-1)
        fg_idx = torch.nonzero(mask_tgt).squeeze()
        fg_coords = sgrid[fg_idx, :] 

        fg2proj, proj2fg, fg2proj_idx, proj2fg_idx = chamfer_python.distChamfer(fg_coords.unsqueeze(0), vert2ds_tgt)

        # project all vertices of the mean shape using source camera
        vert2ds_src = self.renderer.project_points(self.mean_shape, cam_src)

        # for each key point in the source image,
        # search its correspondence on all projected points
        kp_src = kp_src[:, 0:2].cuda()
        kp2proj, _, kp2proj_idx, _ = chamfer_python.distChamfer(kp_src.unsqueeze(0), vert2ds_src)

        kp2proj_idx = kp2proj_idx.squeeze().long()
        proj2fg_idx = proj2fg_idx.squeeze().long()
        kp2fg = fg_coords[proj2fg_idx[kp2proj_idx], :]
        return kp2fg.view(1, kp_num, 2).cpu()

    def predict(self):
        with torch.no_grad():
            outputs = self.model.forward(self.input_imgs)
            # texture flow
            self.tex_flow = outputs['tex_flow']

            # camera
            self.cams = outputs['cam']

            # shape
            self.delta_v = outputs['delta_v']
            if(opts.symmetric):
                delta_v = self.model.symmetrize(self.delta_v)
            self.verts = self.mean_shape + delta_v
            faces = self.model.faces.view(1, -1, 3)
            self.faces = faces.repeat(opts.batch_size, 1, 1)

            # projected mask
            texture_rgba, _, _ = self.renderer.forward(self.verts, self.faces, self.cams)
            pred_mask = texture_rgba[:, 3, :, :]
            self.masks = pred_mask

        if(opts.mode == 'flow'):
            self.k1_to_k2 = self.map_kp_img1_to_img2_flow(self.kps[0], self.tex_flow[0], self.tex_flow[1])
            self.k2_to_k1 = self.map_kp_img1_to_img2_flow(self.kps[1], self.tex_flow[1], self.tex_flow[0])
        elif(opts.mode == 'cam'):
            self.k1_to_k2 = self.map_kp_img1_to_img2_cam(self.kps[0], self.cams[0], self.cams[1], self.masks[1])
            self.k2_to_k1 = self.map_kp_img1_to_img2_cam(self.kps[1], self.cams[1], self.cams[0], self.masks[0])

    def evaluate(self):
        bench_stats = {'kp_errs': [], 'kp_vis': []}
        for iteration, batch in tqdm(enumerate(zip(self.dl_img1, self.dl_img2))):
            # prepare inputs
            batch = collate_pair_batch(batch)
            input_img_tensor = batch['img'].type(torch.FloatTensor)
            for b in range(input_img_tensor.size(0)):
                input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])
            img_tensor = batch['img'].type(torch.FloatTensor)
            mask_tensor = batch['mask'].type(torch.FloatTensor)
            kp_tensor = batch['kp'].type(torch.FloatTensor)
            cam_tensor = batch['sfm_pose'].type(torch.FloatTensor)

            self.input_imgs = input_img_tensor.cuda(device=opts.gpu_id)
            self.imgs = img_tensor.cuda(device=opts.gpu_id)
            self.masks = mask_tensor.cuda(device=opts.gpu_id)
            self.cams = cam_tensor.cuda(device=opts.gpu_id)
            self.kps = kp_tensor
            self.inds = [k.item() for k in batch['inds']]

            self.predict()

            padding_frac = opts.padding_frac
            # The [-1,1] coordinate frame in which keypoints corresponds to:
            #    (1+2*padding_frac)*max_bbox_dim in image coords
            # pt_norm = 2* (pt_img - trans)/((1+2*pf)*max_bbox_dim)
            # err_pt = 2*err_img/((1+2*pf)*max_bbox_dim)
            # err_pck_norm = err_img/max_bbox_dim = err_pt*(1+2*pf)/2
            # so the keypoint error in the canonical fram should be multiplied by:
            err_scaling = (1 + 2 * padding_frac) / 2.0
            kps_gt = self.kps[:, :, 0:2].numpy()
            kps_vis = (self.kps[0, :, 2] * self.kps[1, :, 2]).view(1, self.kps.size(1)).repeat(2, 1)
            kps_pred = torch.cat((self.k2_to_k1, self.k1_to_k2), dim = 0).numpy()
            kps_err = kps_pred - kps_gt
            kps_err = np.sqrt(np.sum(kps_err * kps_err, axis=2)) * err_scaling

            bench_stats['kp_errs'].append(kps_err)
            bench_stats['kp_vis'].append(kps_vis)

            if(opts.visualize):
                # visualize transferred keypoints on target image
                img = self.imgs[0].cpu().permute(1, 2, 0).numpy()
                np_img = np.array(img * 255, dtype = np.uint8).copy()

                kp_num = self.k2_to_k1.squeeze().size(0)
                k2k = (self.k2_to_k1.squeeze() + 1) / 2.0 * 256
                kp_src = self.kps[1]
                kp_src = (kp_src + 1) / 2.0 * 256

                tgt_np = copy.deepcopy(np_img)
                pred_np = copy.deepcopy(np_img)

                img = self.imgs[1].cpu().permute(1, 2, 0).numpy()
                np_img = np.array(img * 255, dtype = np.uint8).copy()

                src_np = copy.deepcopy(np_img)

                kp_tgt = self.kps[0] # 15 x 3
                kp_tgt = (kp_tgt[:, 0:2] + 1) / 2.0 * 256

                for cnt in range(kp_num):
                    if(kps_vis[0, cnt] == 0):
                        continue
                    gt_x = int(k2k[cnt, 0])
                    gt_y = int(k2k[cnt, 1])
                    cv2.circle(pred_np, (gt_x,gt_y), 5, tuple(map(int, color_table[cnt])), -1)

                    gt_x = int(kp_src[cnt, 0])
                    gt_y = int(kp_src[cnt, 1])
                    cv2.circle(src_np, (gt_x,gt_y), 5, tuple(map(int, color_table[cnt])), -1)

                    gt_x = int(kp_tgt[cnt, 0])
                    gt_y = int(kp_tgt[cnt, 1])
                    cv2.circle(tgt_np, (gt_x,gt_y), 5, tuple(map(int, color_table[cnt])), -1)
                target_img = torch.from_numpy(pred_np / 255.0)
                target_img = target_img.permute(2, 0, 1).unsqueeze(0)

                source_img = torch.from_numpy(src_np / 255.0)
                source_img = source_img.permute(2, 0, 1).unsqueeze(0)

                gt_img = torch.from_numpy(tgt_np / 255.0)
                gt_img = gt_img.permute(2, 0, 1).unsqueeze(0)

                self.vis_img = torch.cat((source_img, target_img, gt_img), dim = 0)

                self.save_current_visuals()

            if(iteration == opts.number_pairs):
                break

        bench_stats['kp_errs'] = np.concatenate(bench_stats['kp_errs'])
        bench_stats['kp_vis'] = np.concatenate(bench_stats['kp_vis'])

        n_vis_p = np.sum(bench_stats['kp_vis'], axis=0)
        n_correct_p_pt1 = np.sum(
            (bench_stats['kp_errs'] < 0.1) * bench_stats['kp_vis'], axis=0)
        n_correct_p_pt15 = np.sum(
            (bench_stats['kp_errs'] < 0.15) * bench_stats['kp_vis'], axis=0)
        pck1 = (n_correct_p_pt1 / n_vis_p).mean()
        pck15 = (n_correct_p_pt15 / n_vis_p).mean()
        print(tf_visualizer.green('PCK.1 %.3g, PCK.15 %.3g' % (pck1, pck15)))

    def get_current_visuals(self):
        outputs = {}
        outputs['gt_img'] = self.vis_img[2, :, :, :]
        outputs['source'] = self.vis_img[0, :, :, :]
        outputs['target'] = self.vis_img[1, :, :, :]
        return outputs

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
    tester.evaluate()

if __name__ == '__main__':
    app.run(main)
