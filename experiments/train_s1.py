# ------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Xueting Li.
# ------------------------------------------------------------

# Training Script
# For CUB birds reconstruction, without semantic correspondence constraint.
# Inputs:
#  - single view images
# Outputs:
#  - Predicted camera pose
#  - Reconstructed shape
#  - Reconstructed texture
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..nnutils import geom_utils
from ..nnutils import loss_utils
from ..nnutils import train_utils
from ..nnutils import discriminators
from ..nnutils.smr import SoftRenderer
from ..nnutils import cub_mesh_s1 as mesh_net
from ..nnutils.nmr_pytorch import NeuralRenderer

from ..data import cub as cub_data
from ..utils import image as image_utils
from ..utils import tf_visualizer
from ..utils.tf_visualizer import Visualizer as TfVisualizer

import os
import time
import copy
import numpy as np
import os.path as osp
from absl import app, flags
from collections import OrderedDict

import torch
import torchvision
import soft_renderer as sr
import torchvision.utils as vutils

# Weights:
flags.DEFINE_float('mask_loss_wt', 3.0, 'mask loss weight')
flags.DEFINE_float('grl_wt', .2, 'gradient reversal layer weight')
flags.DEFINE_float('gan_loss_wt', 1., 'adversarial training weight')
flags.DEFINE_float('triangle_reg_wt', 0.15, 'weights to triangle smoothness prior')
flags.DEFINE_float('flatten_reg_wt', 0.0004, 'weights to flatten smoothness prior')
flags.DEFINE_float('deform_reg_wt', 5., 'reg to deformation')
flags.DEFINE_float('ori_reg_wt', 0.4, 'reg to orientation')
flags.DEFINE_float('stop_ori_epoch', 3., 'when to stop usint this constraint')
flags.DEFINE_float('tex_loss_wt', 3.0, 'weights to tex loss')
flags.DEFINE_float('tex_dt_loss_wt', 3.0, 'weights to tex dt loss')
flags.DEFINE_float('tex_cycle_loss_wt', .5, 'weights to tex cycle loss')
# Data:
flags.DEFINE_integer('image_size', 256, 'training image size')
# Model:
flags.DEFINE_string('renderer_type', 'softmax', 'choices are [hard, softmax]')
flags.DEFINE_boolean('use_gan', True, 'If true uses GAN training')
flags.DEFINE_boolean('pred_cam', True, 'If true predicts camera')
flags.DEFINE_boolean('detach_shape', True, 'If true detach shape from the texture branch.')
flags.DEFINE_boolean('detach_cam', True, 'If true detach camera from the texture branch.')
flags.DEFINE_boolean('use_scops', False, 'If true read part segmentations in the loader.')
flags.DEFINE_integer('update_template_freq', 5, 'template update frequency')
flags.DEFINE_integer('axis', 1, 'symmetric axis')


opts = flags.FLAGS

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

class ShapenetTrainer(train_utils.Trainer):
    def define_model(self):
        """
        Define the model

        Args:
            self: (todo): write your description
        """
        opts = self.opts

        # define model
        self.symmetric = opts.symmetric
        img_size = (opts.img_size, opts.img_size)
        self.model = mesh_net.MeshNet(
            img_size, opts, nz_feat=opts.nz_feat,
            axis = opts.axis)
        self.model = self.model.cuda()
        if(opts.multi_gpu):
            self.model = torch.nn.DataParallel(self.model)

        if(opts.use_gan):
            self.discriminator = discriminators.Discriminator(lambda_ = opts.grl_wt,
                                                              img_size = opts.image_size)
            self.discriminator = self.discriminator.cuda()

            if(opts.multi_gpu):
                self.discriminator = torch.nn.DataParallel(self.discriminator)


        if(opts.multi_gpu):
            faces = self.model.module.faces.view(1, -1, 3)
        else:
            faces = self.model.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts.batch_size, 1, 1)

        # define renderers
        self.renderer = SoftRenderer(opts.image_size, opts.renderer_type)
        self.dis_renderer = SoftRenderer(opts.image_size, opts.renderer_type)
        self.hard_renderer = SoftRenderer(opts.image_size, "hard")
        if opts.use_texture:
            self.tex_renderer = SoftRenderer(opts.image_size, opts.renderer_type)
            self.tex_renderer.ambient_light_only()

        self.vis_renderer = NeuralRenderer(opts.image_size)
        self.vis_renderer.ambient_light_only()
        self.vis_renderer.set_bgcolor([1, 1, 1])
        self.vis_renderer.set_light_dir([0, 1, -1], 0.4)

        self.iter_time = 0

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

    def define_criterion(self):
        """
        Define the loss.

        Args:
            self: (todo): write your description
        """
        # shape objectives
        self.mask_loss_fn = loss_utils.neg_iou_loss
        if(opts.multi_gpu):
            verts = self.model.module.get_mean_shape().cpu()
            faces = self.model.module.faces.cpu()
        else:
            verts = self.model.get_mean_shape().cpu()
            faces = self.model.faces.cpu()
        self.laplacian_loss_fn = sr.LaplacianLoss(verts, faces).cuda()
        self.flatten_loss_fn = sr.FlattenLoss(faces).cuda()
        if(opts.multi_gpu):
            self.laplacian_loss_fn = torch.nn.DataParallel(self.laplacian_loss_fn)
            self.flatten_loss_fn = torch.nn.DataParallel(self.flatten_loss_fn)

        # shape constraints
        self.deform_reg_fn = loss_utils.deform_l2reg
        self.ori_reg_fn = loss_utils.sym_reg
        self.gan_loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

        # texture objectives
        if self.opts.use_texture:
            self.texture_loss = loss_utils.PerceptualTextureLoss()
            self.texture_dt_loss_fn = loss_utils.texture_dt_loss
            self.texture_cycle_fn = loss_utils.TexCycle(int(opts.batch_size/opts.gpu_num))
            self.texture_cycle_fn = self.texture_cycle_fn.cuda()
            if(opts.multi_gpu):
                self.texture_cycle_fn = torch.nn.DataParallel(self.texture_cycle_fn)

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

        self.input_imgs = input_img_tensor.cuda()
        self.imgs = img_tensor.cuda()
        self.masks = mask_tensor.cuda()

        if(opts.use_texture):
            # Compute barrier distance transform.
            mask_dts = np.stack([image_utils.compute_dt_barrier(m) for m in mask_tensor])
            dt_tensor = torch.FloatTensor(mask_dts).cuda()
            self.dts_barrier = dt_tensor.unsqueeze(1)

    def forward(self):
        """
        Forward computation

        Args:
            self: (todo): write your description
        """
        opts = self.opts
        outputs = self.model.forward(self.input_imgs)

        # shape
        self.delta_v = outputs['delta_v']
        if(opts.symmetric):
            if(opts.multi_gpu):
                delta_v = self.model.module.symmetrize(self.delta_v)
                self.mean_shape = self.model.module.get_mean_shape()
            else:
                delta_v = self.model.symmetrize(self.delta_v)
                self.mean_shape = self.model.get_mean_shape()
        else:
            delta_v = self.delta_v
        self.pred_vs = self.mean_shape + delta_v

        # camera
        proj_cam = outputs['cam']
        self.proj_cam = proj_cam

        # shape losses
        self.pred_seen, _, _ = self.renderer.forward(self.pred_vs, self.faces, proj_cam)
        self.mask_pred_seen = self.pred_seen[:,3,:,:]
        self.mask_loss = self.mask_loss_fn(self.mask_pred_seen, self.masks)

        self.triangle_loss = self.laplacian_loss_fn(self.pred_vs).mean()
        self.flatten_loss = self.flatten_loss_fn(self.pred_vs).mean()
        self.deform_loss = self.deform_reg_fn(self.delta_v)
        self.ori_loss = self.ori_reg_fn(self.pred_vs)

        # texture losses
        if(opts.use_texture):
            self.tex_flow = outputs['tex_flow']
            self.uvimage_pred = outputs['uvimage_pred']
            self.tex = geom_utils.sample_textures(self.tex_flow, self.imgs)
            self.tex = self.tex.contiguous()

            bs, fs, ts, _, _ = self.tex.size()
            self.tex = self.tex.view(bs, fs, -1, 3)
            texture_rgba, p2f_info, _ = self.tex_renderer.forward(self.pred_vs.detach(), self.faces, proj_cam.detach(), self.tex)
            self.texture_pred = texture_rgba[:,0:3,:,:]
            self.tex_loss = self.texture_loss(self.texture_pred, self.imgs, self.masks, self.mask_pred_seen)
            self.tex_dt_loss = self.texture_dt_loss_fn(self.tex_flow, self.dts_barrier)

            # texture cycle loss
            _, _, aggr_info = self.hard_renderer(self.pred_vs.detach(), self.faces, proj_cam.detach())
            aggr_info = aggr_info[:, 1, :, :].view(bs, -1)

            tex_cycle_loss, self.avg_flow = self.texture_cycle_fn(self.tex_flow, p2f_info.detach(), aggr_info.detach())
            # The mean is used to collect loss from different GPUs
            self.tex_cycle_loss = torch.mean(tex_cycle_loss)
            self.p2f_info = p2f_info

        if(opts.use_gan):
            # render at unobserved view
            angles = np.random.randint(0, 180, size=bs)
            random_cams = geom_utils.rotate_cam(proj_cam.detach(), angles)
            pred_unseen, _, _ = self.dis_renderer.forward(self.pred_vs, self.faces, random_cams)
            self.mask_pred_unseen = pred_unseen[:,3,:,:]

            pred = torch.cat((self.pred_seen.detach(), pred_unseen))
            gan_labels = torch.cat((torch.ones(self.pred_seen.shape[0]),
                                    torch.zeros(pred_unseen.shape[0])), dim = 0)
            gan_labels = gan_labels.cuda()
            gan_preds = self.discriminator(pred[:,3,:,:].unsqueeze(1))
            self.gan_loss = self.gan_loss_fn(gan_preds.squeeze(), gan_labels)

        # add up all losses
        # shape
        self.total_loss = self.mask_loss * opts.mask_loss_wt
        self.total_loss += self.triangle_loss * opts.triangle_reg_wt
        self.total_loss += self.flatten_loss * opts.flatten_reg_wt
        if(self.curr_epoch < opts.stop_ori_epoch):
            # constrain prediction to be symmetric on the given axis
            self.total_loss += self.ori_loss * opts.ori_reg_wt
        if(self.curr_epoch > opts.update_template_freq):
            # constrain prediction from deviating from template
            self.total_loss += self.deform_loss * opts.deform_reg_wt

        # texture
        if(opts.use_texture):
            self.total_loss += self.tex_loss * opts.tex_loss_wt
            self.total_loss += self.tex_dt_loss * opts.tex_dt_loss_wt
            self.total_loss += self.tex_cycle_loss * opts.tex_cycle_loss_wt

        # GAN
        if(opts.use_gan):
            self.total_loss += self.gan_loss * opts.gan_loss_wt

    def get_current_visuals(self):
        """
        Get the current image images

        Args:
            self: (todo): write your description
        """
        vis_dict = {}

        # UV maps
        if self.opts.use_texture:
            uv_flows = self.uvimage_pred
            uv_flows = uv_flows.permute(0, 2, 3, 1)
            uv_images = torch.nn.functional.grid_sample(self.imgs, uv_flows)

            vis_dict['uv_images'] = uv_images

        # mask
        vis_dict['mask_pred'] = self.mask_pred_seen.unsqueeze(1)
        nb, nf, _, nc = self.tex.size()
        tex = self.tex.detach().view(nb, nf, opts.tex_size, opts.tex_size, nc).unsqueeze(4).repeat(1, 1, 1, 1, opts.tex_size, 1)
        vis_dict['mask_gt'] = self.masks.unsqueeze(1)

        # image
        vis_dict['image_pred'] = self.vis_renderer(self.pred_vs.detach(), self.faces, self.proj_cam.detach(), tex)
        vis_dict['image_gt'] = self.imgs * self.masks.unsqueeze(1).repeat(1, 3, 1, 1)

        # instance mesh
        if(self.opts.use_texture):
            mesh_ = sr.Mesh(self.pred_vs[0], self.faces[0], self.tex[0].view(self.faces.size(1),-1,3))
        else:
            mesh_ = sr.Mesh(self.pred_vs[0], self.faces[0])
        vis_dict['mesh'] = mesh_

        # template mesh
        if(opts.multi_gpu):
            template_mesh_ = sr.Mesh(self.model.module.get_mean_shape(), self.faces[0])
        else:
            template_mesh_ = sr.Mesh(self.model.get_mean_shape(), self.faces[0])
        vis_dict['template_mesh'] = template_mesh_

        return vis_dict

    def get_current_scalars(self):
        """
        Returns a dictionary of the scalars.

        Args:
            self: (todo): write your description
        """
        opts = self.opts
        sc_dict = OrderedDict([
            ('smoothed_total_loss', self.smoothed_total_loss),
            ('total_loss', self.total_loss),
            ('mask_loss', self.mask_loss),
            ('tri_loss', self.triangle_loss),
            ('flatten_loss', self.flatten_loss),
            ('deform_loss', self.deform_loss),
            ('ori_loss', self.ori_loss),
            ('lr', self.optimizer.param_groups[0]['lr']),
            ('iter_time', self.iter_time),
        ])
        if opts.use_texture:
            sc_dict['tex_loss'] = self.tex_loss
            sc_dict['tex_dt_loss'] = self.tex_dt_loss
            sc_dict['tex_cycle_loss'] = self.tex_cycle_loss

        return sc_dict

    '''Overwrite train function for template update.'''
    def train(self):
        """
        Training function.

        Args:
            self: (todo): write your description
        """
        opts = self.opts
        self.visualizer = TfVisualizer(opts)
        self.smoothed_total_loss = 0

        visualizer = self.visualizer
        total_steps = 0
        optim_steps = 0
        dataset_size = len(self.dataloader)

        for epoch in range(opts.num_pretrain_epochs, opts.num_epochs):
            epoch_iter = 0
            self.curr_epoch = epoch
            for i, batch in enumerate(self.dataloader):
                self.iteration_num += 1
                self.adjust_learning_rate(self.optimizer)
                t_init = time.time()
                self.set_input(batch)
                t_batch = time.time()

                if not self.invalid_batch:
                    optim_steps += 1
                    self.optimizer.zero_grad()

                    start_time = time.time()
                    self.forward()
                    self.smoothed_total_loss = self.smoothed_total_loss*0.99 + 0.01*self.total_loss
                    t_forw = time.time()
                    self.total_loss.backward()
                    t_backw = time.time()
                    if optim_steps % opts.optim_bs == 0:
                        self.optimizer.step()

                    end_time = time.time()
                    self.iter_time = end_time - start_time
                    t_opt = time.time()

                total_steps += 1
                epoch_iter += 1

                if opts.display_visuals and (total_steps % opts.display_freq == 0):
                    iter_end_time = time.time()
                    vis_dict = self.get_current_visuals()
                    for k,v in vis_dict.items():
                        if('mesh' in k):
                            v.save_obj(os.path.join(self.vis_dir,'{}.obj'.format(k)), save_texture=True)
                        else:
                            vutils.save_image(v, os.path.join(self.vis_dir, k + '.png'))
                    print(tf_visualizer.green("Visualization saved at {}.".format(self.vis_dir)))

                if opts.print_scalars and (total_steps % opts.print_freq == 0):
                    scalars = self.get_current_scalars()
                    visualizer.print_current_scalars(epoch, epoch_iter, scalars)

                if total_steps % opts.save_latest_freq == 0:
                    print(tf_visualizer.green('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps)))
                    self.save('latest')

                if total_steps == opts.num_iter:
                    return

            # update template
            if((epoch+1) % opts.update_template_freq == 0):
                print(tf_visualizer.green('Updating template...'))
                self.feat = torch.zeros(opts.batch_size, opts.z_dim)
                self.feat = self.feat.cuda()

                # compute average encoder features
                for i, batch in enumerate(self.dataloader):
                    self.set_input(batch)

                    with torch.no_grad():
                        outputs = self.model(self.input_imgs)
                    self.feat += outputs['feat']
                self.feat = self.feat / (i + 1)
                self.feat = torch.mean(self.feat, dim=0).unsqueeze(0)

                # feed averaged features into the shape decoder
                if(opts.multi_gpu):
                    with torch.no_grad():
                        delta_v = self.model.module.shape_predictor(self.feat)
                    self.model.module.mean_v += delta_v.squeeze()
                else:
                    with torch.no_grad():
                        delta_v = self.model.shape_predictor(self.feat)
                    self.model.mean_v += delta_v.squeeze()
                print(tf_visualizer.green('Template updated.'))

            if (epoch+1) % opts.save_epoch_freq == 0:
                print(tf_visualizer.green('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps)))
                self.save('latest')
                self.save(epoch+1)

def main(_):
    """
    Main function.

    Args:
        _: (int): write your description
    """
    torch.manual_seed(0)
    trainer = ShapenetTrainer(opts)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)
