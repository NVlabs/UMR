# -----------------------------------------------------------
# Code adapted from: 
# https://github.com/akanazawa/cmr/blob/master/nnutils/test_utils.py
#
# MIT License
# Copyright (c) 2018 akanazawa
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------

# Generic Testing Utils.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
import os.path as osp
import time
import imageio
import pdb
import numpy as np
from absl import flags

import torchvision.utils as vutils
import soft_renderer as sr

#-------------- flags -------------#
#----------------------------------#
## Flags for training
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

flags.DEFINE_string('name', 'exp_name', 'Experiment Name')
flags.DEFINE_string('cache_dir', cache_path, 'Cachedir')
flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')

flags.DEFINE_integer('num_train_epoch', 0, 'Number of training iterations')
flags.DEFINE_integer('num_pretrain_epochs', 0, 'If >0, we will pretain from an existing saved model.')
flags.DEFINE_integer('batch_size', 2, 'Size of minibatches')
flags.DEFINE_integer('workers', 8, 'dataloader worker number')

## Flags for logging and snapshotting
flags.DEFINE_string('checkpoint_dir', osp.join(cache_path, 'snapshots'),
                    'Directory where networks are saved')
flags.DEFINE_string(
    'results_vis_dir', osp.join(cache_path, 'results_vis'),
    'Directory where intermittent results will be saved')
flags.DEFINE_string(
    'results_eval_dir', osp.join(cache_path, 'evaluation'),
    'Directory where evaluation results will be saved')

flags.DEFINE_boolean('save_visuals', False, 'Whether to save intermittent visuals')
flags.DEFINE_integer('visuals_freq', 50, 'Save visuals every few forward passes')
flags.DEFINE_integer('max_eval_iter', 0, 'Maximum evaluation iterations. 0 => 1 epoch.')


#--------- testing class ----------#
#----------------------------------#
class Tester():
    def __init__(self, opts):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            opts: (todo): write your description
        """
        self.opts = opts
        self.vis_iter = 0
        self.gpu_id = opts.gpu_id
        torch.cuda.set_device(opts.gpu_id)

        self.Tensor = torch.cuda.FloatTensor if (self.gpu_id is not None) else torch.Tensor
        self.invalid_batch = False #the trainer can optionally reset this every iteration during set_input call
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        log_file = os.path.join(self.save_dir, 'opts_testing.log')
        with open(log_file, 'w') as f:
            for k in dir(opts):
                f.write('{}: {}\n'.format(k, opts.__getattr__(k)))

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, network_dir=None):
        """
        Load network from disk.

        Args:
            self: (todo): write your description
            network: (todo): write your description
            network_label: (str): write your description
            epoch_label: (todo): write your description
            network_dir: (str): write your description
        """
        if(epoch_label > 0):
            save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        else:
            save_filename = '{}_net_latest.pth'.format(network_label)
        if network_dir is None:
            network_dir = self.save_dir
        save_path = os.path.join(network_dir, save_filename)
        network.load_state_dict(torch.load(save_path))
        print('===> Loaded pretrained network from %s.'%save_path)
        return network

    def load_my_state_dict(self, resume_dir):
        """
        Loads the model dictionary from the dir.

        Args:
            self: (todo): write your description
            resume_dir: (str): write your description
        """
        saved_state_dict = torch.load(resume_dir)
        unwanted_keys = {"uv_sampler"}

        # only copy the params that exist in current model (caffe-like)
        new_params = self.model.state_dict().copy()
        for name, param in new_params.items():
            if name in saved_state_dict and param.size() == saved_state_dict[name].size() and name not in unwanted_keys:
                new_params[name].copy_(saved_state_dict[name])
        self.model.load_state_dict(new_params)
        print("Loaded model from {}.".format(resume_dir))

    def save_current_visuals(self, unsup=False):
        """
        Saves all visual visualisation images.

        Args:
            self: (todo): write your description
            unsup: (todo): write your description
        """
        def norm_ip(img, min, max):
            """
            Assigns an array.

            Args:
                img: (todo): write your description
                min: (float): write your description
                max: (int): write your description
            """
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        if(unsup):
            visuals = self.get_current_visuals_unsup()
        else:
            visuals = self.get_current_visuals()
        imgs_dir = osp.join(self.opts.results_vis_dir, 'vis_iter_{}'.format(self.vis_iter))
        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir)
        for k in visuals:
            if("mesh" in k):
                meshes = visuals[k]
                verts = meshes.vertices
                faces = meshes.faces
                tex = meshes.textures
                for cnt in range(verts.size(0)):
                    mesh_path = osp.join(imgs_dir, k + '_' + str(cnt) + '.obj')
                    mesh_ = sr.Mesh(verts[cnt], faces[cnt], tex[cnt])
                    mesh_.save_obj(mesh_path, save_texture=self.opts.use_texture)
            elif("GIF" in k):
                for cnt in range(self.opts.sample_num):
                    vis_image = torch.cat((visuals['gt_imgs'][cnt].unsqueeze(0).cpu(), visuals[k][cnt].cpu()), dim = 0)
                    nrow = vis_image.size(0)
                    vutils.save_image(vis_image, os.path.join(imgs_dir, "{}.png".format(cnt)), normalize=True, scale_each=True, nrow=nrow)
            else:
                img_path = osp.join(imgs_dir, k + '.png')
                vutils.save_image(visuals[k], img_path, normalize=False)
        self.vis_iter += 1

    def define_model(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_dataset(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def set_input(self, batch):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_current_visuals(self):
        """
        Return a list of visual visual visual visual values

        Args:
            self: (todo): write your description
        """
        return {}

    def init_testing(self):
        """
        Initialize the underlying dataset.

        Args:
            self: (todo): write your description
        """
        self.init_dataset()
        self.define_model()

    def test(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError
