# ------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Xueting Li.
# ------------------------------------------------------------

# Testing Script for quantitative evaluations
# Inputs:
#  - single view images
# Evaluation metric:
#  - Intersection-over-Union between projected masks and GT masks
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

from ..utils import tf_visualizer
from ..data import cub as cub_data

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
flags.DEFINE_integer('axis', 1, 'symmetric axis')
flags.DEFINE_boolean('use_scops', False, 'If true predicts camera')
flags.DEFINE_string('model_path', 'unsup-mesh/cachedir/snapshots/cub_net/pred_net_latest.pth', 'model path')

opts = flags.FLAGS

class ShapenetTester(test_utils.Tester):
    def define_model(self):
        opts = self.opts

        # define model
        img_size = (opts.image_size, opts.image_size)
        self.model = mesh_net.MeshNet(
            img_size, opts, nz_feat=opts.nz_feat,
            axis = opts.axis)
        self.load_my_state_dict(os.path.join(opts.model_path))
        self.model = self.model.cuda(device=opts.gpu_id)
        self.model.eval()
        self.mean_shape = self.model.get_mean_shape().unsqueeze(0)

        # define renderer
        self.renderer = SoftRenderer(opts.image_size, 'softmax')

        return

    def init_dataset(self):
        opts = self.opts
        self.data_module = cub_data
        self.dataloader = self.data_module.data_loader(opts)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def evaluate(self):
        mask_iou = 0
        ious = []
        for i, batch in tqdm(enumerate(self.dataloader)):
            input_img_tensor = batch['img'].type(torch.FloatTensor)
            for b in range(input_img_tensor.size(0)):
                input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])
            img_tensor = batch['img'].type(torch.FloatTensor)
            mask_tensor = batch['mask'].type(torch.FloatTensor)

            self.input_imgs = input_img_tensor.cuda(device=opts.gpu_id)
            self.imgs = img_tensor.cuda(device=opts.gpu_id)
            self.masks = mask_tensor.cuda(device=opts.gpu_id)

            with torch.no_grad():
                outputs = self.model.forward(self.input_imgs)
                # shape
                self.delta_v = outputs['delta_v']
                if(opts.symmetric):
                    delta_v = self.model.symmetrize(self.delta_v)
                self.verts = self.mean_shape + delta_v
                faces = self.model.faces.view(1, -1, 3)
                self.faces = faces.repeat(opts.batch_size, 1, 1)

                # camera
                pred_cams = outputs['cam']

                # projected mask
                texture_rgba, _, _ = self.renderer.forward(self.verts, self.faces, pred_cams)
                pred_mask = texture_rgba[:, 3, :, :]

                # compute IoU
                mask_gt = self.masks.cpu().view(self.masks.size(0), -1).numpy()
                mask_pred = pred_mask.cpu().view(self.masks.size(0), -1).numpy()
                intersection = mask_gt * mask_pred
                union = mask_gt + mask_pred - intersection
                iou = intersection.sum(1) / union.sum(1)
                ious.append(iou)

                mask_iou += np.mean(iou)

        iou_concat = np.concatenate(ious)
        print(tf_visualizer.green("Average mask IoU: {:.4f}.".format(iou_concat.mean())))

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
