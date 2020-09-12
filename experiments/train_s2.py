# ------------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Xueting Li.
# ------------------------------------------------------------

# Training Script
# For CUB birds reconstruction.
# With semantic correspondence constraints.
# Inputs:
#  - single view images
# Outputs:
#  - Predicted viewpoints
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
from ..nnutils import cub_mesh as mesh_net
from ..nnutils.nmr_pytorch import NeuralRenderer

from ..data import cub as cub_data

from ..utils import tf_visualizer
from ..utils import image as image_utils
from ..utils.tf_visualizer import Visualizer as TfVisualizer

import os
import cv2
import time
import random
import numpy as np
import os.path as osp
from absl import app, flags
from collections import OrderedDict

import torch
import torchvision
import soft_renderer as sr
import torchvision.utils as vutils

# Weights:
flags.DEFINE_float('mask_loss_wt', 2.5, 'mask loss weight')
flags.DEFINE_float('grl_wt', .2, 'gradient reversal layer weight')
flags.DEFINE_float('gan_loss_wt', 1., 'adversarial training weight')
flags.DEFINE_float('triangle_reg_wt', 0.15, 'weights to triangle smoothness prior')
flags.DEFINE_float('flatten_reg_wt', 0.0005, 'weights to flatten smoothness prior')
flags.DEFINE_float('tex_loss_wt', 3.0, 'weights to tex loss')
flags.DEFINE_float('tex_dt_loss_wt', 3.0, 'weights to tex dt loss')
flags.DEFINE_float('tex_cycle_loss_wt', 1.0, 'weights to tex cycle loss')
flags.DEFINE_float('ent_loss_wt', 0.05, 'Cam diversity loss wt.')
flags.DEFINE_float('prob_loss_wt', 5.0, 'probability-based semantic loss weight.')
flags.DEFINE_float('vertex_loss_wt', 10.0, 'vertex-based semantic loss weight.')
flags.DEFINE_float('deform_reg_wt', 1., 'reg to deformation')
# Data:
flags.DEFINE_integer('image_size', 256, 'training image size')
flags.DEFINE_string('stemp_path', 'unsup-mesh/cachedir/cub/scops/', 'path to semantic template.')
flags.DEFINE_boolean('use_scops', True, 'if True, use SCOPS')
flags.DEFINE_string('dataset', 'cub', 'cub|pascal|imnet')
# Model:
flags.DEFINE_string('renderer_type', 'softmax', 'choices are [hard, softmax]')
flags.DEFINE_boolean('use_gan', True, 'If true uses GAN training')
flags.DEFINE_boolean('pred_cam', True, 'If true predicts camera')
flags.DEFINE_boolean('detach_shape', True, 'If true detach shape from the texture branch.')
flags.DEFINE_boolean('detach_cam', True, 'If true detach camera from the texture branch.')
flags.DEFINE_integer('axis', 1, 'symmetric axis')


opts = flags.FLAGS

class ShapenetTrainer(train_utils.Trainer):
    def define_model(self):
        opts = self.opts

        # define model
        img_size = (opts.img_size, opts.img_size)
        self.model = mesh_net.MeshNet(
                     img_size, opts, nz_feat=opts.nz_feat,
                     axis = opts.axis,
                     temp_path = opts.stemp_path)
        self.model = self.model.cuda()

        if(opts.use_gan):
            self.discriminator = discriminators.Discriminator(in_dim = 3,
                                                              lambda_ = opts.grl_wt,
                                                              img_size = opts.image_size)
            self.discriminator = self.discriminator.cuda()
            if(opts.multi_gpu):
                self.discriminator = torch.nn.DataParallel(self.discriminator)

        faces = self.model.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts.batch_size, 1, 1)
        self.mean_shape = self.model.get_mean_shape()
        if(opts.multi_gpu):
            self.model = torch.nn.DataParallel(self.model)

        # define renderers
        self.renderer = SoftRenderer(opts.image_size, opts.renderer_type)
        self.dis_renderer = SoftRenderer(opts.image_size, opts.renderer_type)
        self.dis_renderer.ambient_light_only()
        if opts.use_texture:
            self.tex_renderer = SoftRenderer(opts.image_size, opts.renderer_type)
            self.tex_renderer.ambient_light_only()

        self.vis_renderer = NeuralRenderer(opts.image_size)
        self.vis_renderer.ambient_light_only()
        self.vis_renderer.set_light_dir([0, 1, -1], 0.4)

        self.iter_time = 0
        self.random_imgs = None

        return

    def init_dataset(self):
        opts = self.opts
        self.data_module = cub_data
        self.dataloader = self.data_module.data_loader(opts)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def define_criterion(self):
        # shape objectives
        self.mask_loss_fn = loss_utils.MultiMaskLoss(opts.image_size,
                                                       opts.renderer_type,
                                                       opts.num_hypo_cams)
        self.mask_loss_fn = torch.nn.DataParallel(self.mask_loss_fn)

        verts = self.mean_shape.cpu()
        faces = self.faces[0].cpu()

        self.laplacian_loss_fn = sr.LaplacianLoss(verts, faces).cuda()
        self.flatten_loss_fn = sr.FlattenLoss(faces).cuda()
        if(opts.multi_gpu):
            self.laplacian_loss_fn = torch.nn.DataParallel(self.laplacian_loss_fn)
            self.flatten_loss_fn = torch.nn.DataParallel(self.flatten_loss_fn)

        self.gan_loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        self.deform_reg_fn = loss_utils.deform_l2reg

        # texture objectives
        self.texture_loss_fn = loss_utils.MultiTextureLoss(int(opts.batch_size/opts.gpu_num))
        self.texture_loss_fn = torch.nn.DataParallel(self.texture_loss_fn)

        # semantic correspondence constraints
        self.corr_loss_fn = loss_utils.CorrLossChamfer(opts.stemp_path, opts.image_size)
        
        if(opts.multi_gpu):
            self.part_loss_fn = loss_utils.part_matching_loss(opts.stemp_path, self.model.module.uv_sampler[0].unsqueeze(0),
                                                              self.model.module.texture_predictor.num_sym_faces,
                                                              batch_size = int(opts.batch_size/opts.gpu_num))
        else:
            self.part_loss_fn = loss_utils.part_matching_loss(opts.stemp_path, self.model.uv_sampler[0].unsqueeze(0),
                                                              self.model.texture_predictor.num_sym_faces,
                                                              batch_size = int(opts.batch_size/opts.gpu_num))
                                                              
        self.part_loss_fn = self.part_loss_fn.cuda()
        self.part_loss_fn = torch.nn.DataParallel(self.part_loss_fn)

    def set_input(self, batch):
        opts = self.opts

        input_img_tensor = batch['img'].type(torch.FloatTensor)
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])
        img_tensor = batch['img'].type(torch.FloatTensor)
        mask_tensor = batch['mask'].type(torch.FloatTensor)
        part_seg_tensor = batch['part_prob'].type(torch.FloatTensor)
        self.head_points = batch['head_points'].cuda()
        self.belly_points = batch['belly_points'].cuda()
        self.back_points = batch['back_points'].cuda()
        self.neck_points = batch['neck_points'].cuda()

        self.input_imgs = input_img_tensor.cuda()
        self.imgs = img_tensor.cuda()
        self.masks = mask_tensor.cuda()
        if(self.random_imgs is None):
            self.random_imgs = self.imgs * self.masks.unsqueeze(1).repeat(1, 3, 1, 1)

        # normalize part segmentations
        self.part_segs = part_seg_tensor.permute(0, 3, 1, 2).cuda()
        for ch in range(5):
            part_prob_slice = self.part_segs[:, ch, :, :]
            part_prob_slice = part_prob_slice * self.masks
            k, _ = torch.max(part_prob_slice.view(part_prob_slice.size(0), -1), dim = 1)
            self.part_segs[:, ch, :, :] = part_prob_slice / k.view(-1, 1, 1)

        if(opts.use_texture):
            # Compute barrier distance transform.
            mask_dts = np.stack([image_utils.compute_dt_barrier(m) for m in mask_tensor])
            dt_tensor = torch.FloatTensor(mask_dts).cuda()
            # B x 1 x N x N
            self.dts_barrier = dt_tensor.unsqueeze(1)

    def forward(self):
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

        # camera losses
        self.proj_cam = outputs['cam'].detach()
        cams_all_hypo = outputs['cam_hypotheses']
        self.cams_all_hypo = cams_all_hypo
        self.cam_probs = outputs['cam_probs']
        self.cam_div_loss = -1*(torch.log(self.cam_probs + 1E-9)*self.cam_probs).sum(1).mean()

        # shape losses
        self.mask_loss, self.mask_all_hypo = self.mask_loss_fn(self.pred_vs, self.faces,
                                             self.cams_all_hypo, self.cam_probs, self.masks)
        self.mask_loss = self.mask_loss.mean() # collect loss from multiple GPUs

        self.triangle_loss = self.laplacian_loss_fn(self.pred_vs).mean()
        self.flatten_loss = self.flatten_loss_fn(self.pred_vs).mean()
        self.deform_loss = self.deform_reg_fn(self.delta_v)

        # texture losses
        if(opts.use_texture):
            self.tex_flow = outputs['tex_flow']
            self.uvimage_pred = outputs['uvimage_pred']
            self.tex = geom_utils.sample_textures(self.tex_flow, self.imgs)
            self.tex = self.tex.contiguous()

            bs, fs, ts, _, _ = self.tex.size()
            self.tex = self.tex.view(bs, fs, -1, 3)

            # multiple camera hypotheses
            # detach camera from the texture branch
            cams_all_hypo = outputs['cam_hypotheses'].detach()
            cam_probs = outputs['cam_probs'].detach()
            self.tex_loss, self.tex_dt_loss, self.tex_cycle_loss, self.texture_pred = self.texture_loss_fn(self.pred_vs.detach(), self.faces, cams_all_hypo, cam_probs, self.proj_cam, self.imgs, self.masks, self.mask_all_hypo, self.tex, self.tex_flow, self.dts_barrier)

            # collect loss from different GPUs
            self.tex_loss = self.tex_loss.mean()
            self.tex_dt_loss = self.tex_dt_loss.mean()
            self.tex_cycle_loss = self.tex_cycle_loss.mean()

        if(opts.use_gan):
            # render at observed view
            angles = np.random.randint(0, 180, size=bs)
            random_cams = geom_utils.rotate_cam(self.proj_cam.detach(), angles)

            pred_unseen, _, _ = self.dis_renderer.forward(self.pred_vs, self.faces, random_cams, self.tex.detach())
            pred = torch.cat((self.random_imgs, pred_unseen[:, 0:3]))

            gan_labels = torch.cat((torch.ones(self.random_imgs.shape[0]),
                                    torch.zeros(self.pred_vs.shape[0])), dim = 0)
            gan_labels = gan_labels.cuda()
            gan_preds = self.discriminator(pred)
            self.gan_loss = self.gan_loss_fn(gan_preds.squeeze(), gan_labels)

        # update random images
        self.random_imgs = self.imgs * self.masks.unsqueeze(1).repeat(1, 3, 1, 1)

        # add all losses
        # shape
        self.total_loss = self.mask_loss * opts.mask_loss_wt
        self.total_loss += self.triangle_loss * opts.triangle_reg_wt
        self.total_loss += self.flatten_loss * opts.flatten_reg_wt
        self.total_loss += self.deform_loss * opts.deform_reg_wt

        # texture
        if(opts.use_texture):
            self.total_loss += self.tex_loss * opts.tex_loss_wt
            self.total_loss += self.tex_dt_loss * opts.tex_dt_loss_wt
            self.total_loss += self.tex_cycle_loss * opts.tex_cycle_loss_wt

        # GAN
        if(opts.use_gan):
            self.total_loss += self.gan_loss * opts.gan_loss_wt
            # gan acc
            gan_preds = torch.sigmoid(gan_preds)
            gan_preds_labels = (gan_preds.squeeze() >= 0.5).float()
            self.acc_gan = torch.sum(gan_preds_labels == gan_labels).float() / (gan_labels.size(0))

        # camera
        self.total_loss += self.cam_div_loss * opts.ent_loss_wt

        # semantic correspondence losses
        part_loss, self.part_projs = self.part_loss_fn(self.pred_vs, self.faces,
                                                       self.proj_cam, self.part_segs)
        self.part_loss = torch.mean(part_loss)
        self.total_loss += self.part_loss * opts.prob_loss_wt

        mean_shape = self.mean_shape.unsqueeze(0).repeat(self.pred_vs.size(0), 1, 1)
        if(opts.multiple_cam_hypo):
            self.corr_loss = 0
            cams_all_hypo_flat = self.cams_all_hypo.view(-1, 7)
            mean_shape_repeat = mean_shape.unsqueeze(1).repeat(1, 8, 1, 1).view(-1, mean_shape.size(1), 3)
            head_points = self.head_points.unsqueeze(1).repeat(1, 8, 1, 1).view(-1, self.head_points.size(1), 2)
            belly_points = self.belly_points.unsqueeze(1).repeat(1, 8, 1, 1).view(-1, self.belly_points.size(1), 2)
            back_points = self.back_points.unsqueeze(1).repeat(1, 8, 1, 1).view(-1, self.back_points.size(1), 2)
            neck_points = self.neck_points.unsqueeze(1).repeat(1, 8, 1, 1).view(-1, self.neck_points.size(1), 2)
            corr_loss = self.corr_loss_fn(head_points, belly_points, back_points, neck_points, mean_shape_repeat, cams_all_hypo_flat, avg = False)

            corr_loss = (corr_loss.view(bs, 8) * cam_probs).sum(dim = 1)
            self.corr_loss = corr_loss.mean()
        # parallel loss
        self.total_loss += self.corr_loss * opts.vertex_loss_wt

    def get_current_visuals(self):
        vis_dict = {}
        
        nb, nf, _, nc = self.tex.size()
        tex = self.tex.detach().view(nb, nf, opts.tex_size, opts.tex_size, nc).unsqueeze(4).repeat(1, 1, 1, 1, opts.tex_size, 1)
        vis_dict['image_pred'] = self.vis_renderer(self.pred_vs.detach(), self.faces, self.proj_cam.detach(), tex)
        vis_dict['mask_pred'] = self.vis_renderer(self.pred_vs.detach(), self.faces, self.proj_cam.detach()).unsqueeze(1)

        uv_flows = self.uvimage_pred.detach()
        uv_flows = uv_flows.permute(0, 2, 3, 1)
        uv_images = torch.nn.functional.grid_sample(self.imgs, uv_flows)

        vis_dict['uv_images'] = uv_images

        vis_dict['mask_gt'] = self.masks.unsqueeze(1)
        vis_dict['image_gt'] = self.imgs

        for cnt in range(5):
            mesh_ = sr.Mesh(self.pred_vs[cnt].detach(), self.faces[cnt], self.tex[cnt].view(self.faces.size(1),-1,3).detach())
            vis_dict['mesh_'+str(cnt)] = mesh_

        # visualize part projects
        for cnt in range(len(self.part_projs)):
            vis_dict['part_render_' + str(cnt)] = self.part_projs[cnt]

        # visualize gt part projects
        for cnt in range(4):
            vis_dict['gt_part_render_' + str(cnt)] = self.part_segs[:, cnt + 1].unsqueeze(1)

        mean_shape = self.mean_shape.unsqueeze(0).repeat(self.pred_vs.size(0), 1, 1)
        _, vert2ds = self.corr_loss_fn(self.head_points, self.belly_points, self.back_points, self.neck_points, mean_shape, self.proj_cam.detach())
        vert2ds = (vert2ds + 1.0) / 2 * opts.image_size
        self.center_imgs = torch.zeros(self.imgs.size())
        for cnt in range(self.imgs.size(0)):
            img_ = self.imgs[cnt]
            img_ = img_.cpu().permute(1,2,0).numpy()
            img_ = img_ * 255.0
            img_ = np.array(img_, dtype = np.uint8).copy()

            vert2d = vert2ds[cnt]
            for ccnt in range(vert2d.size(0)):
                x = int(vert2d[ccnt, 0].cpu().detach().numpy())
                y = int(vert2d[ccnt, 1].cpu().detach().numpy())
                head_num = self.corr_loss_fn.head_num 
                belly_num = self.corr_loss_fn.belly_num 
                # we only visualize points from the head and belly parts
                # since they're most informative.
                if(ccnt < head_num):
                    cv2.drawMarker(img_, (x,y), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize = 5, thickness=2)
                elif(ccnt < head_num + belly_num):
                    cv2.drawMarker(img_, (x,y), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize = 5, thickness=2)

            head_points  = (self.head_points[cnt] + 1) / 2.0 * 256.0
            belly_points = (self.belly_points[cnt] + 1) / 2.0 * 256.0
            for ccnt in range(head_points.size(0)):
                x = int(head_points[ccnt, 0].cpu().numpy())
                y = int(head_points[ccnt, 1].cpu().numpy())
                cv2.drawMarker(img_, (x,y), (0,255,255), markerType=cv2.MARKER_CROSS, markerSize = 5, thickness=2)

            for ccnt in range(belly_points.size(0)):
                x = int(belly_points[ccnt, 0].cpu().numpy())
                y = int(belly_points[ccnt, 1].cpu().numpy())
                cv2.drawMarker(img_, (x,y), (255,255,0), markerType=cv2.MARKER_CROSS, markerSize = 5, thickness=2)

            self.center_imgs[cnt] = torch.from_numpy(img_/255.0).permute(2,0,1).unsqueeze(0).float()

        vis_dict['part_cons_vis'] = self.center_imgs
        return vis_dict

    def get_current_scalars(self):
        opts = self.opts
        sc_dict = OrderedDict([
            ('smoothed_total_loss', self.smoothed_total_loss),
            ('total_loss', self.total_loss),
            ('mask_loss', self.mask_loss),
            ('tri_loss', self.triangle_loss),
            ('flatten_loss', self.flatten_loss),
            ('deform_loss', self.deform_loss),
            ('lr', self.optimizer.param_groups[0]['lr']),
            ('cam_div_loss', self.cam_div_loss),
            ('part_loss', self.part_loss * opts.prob_loss_wt),
            ('corr_loss', self.corr_loss),
            ('iter_time', self.iter_time),
        ])
        if opts.use_texture:
            sc_dict['tex_loss'] = self.tex_loss
            sc_dict['tex_dt_loss'] = self.tex_dt_loss
            sc_dict['tex_cycle_loss'] = self.tex_cycle_loss

        return sc_dict

    def train(self):
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
                    start_time = time.time()
                    self.optimizer.zero_grad()

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
                            v.save_obj(osp.join(self.vis_dir,'{}.obj'.format(k)), save_texture=True)
                        else:
                            vutils.save_image(v, osp.join(self.vis_dir, k + '.png'))
                    del vis_dict
                    print(tf_visualizer.green("Visualization saved at {}.".format(self.vis_dir)))

                if opts.print_scalars and (total_steps % opts.print_freq == 0):
                    scalars = self.get_current_scalars()
                    visualizer.print_current_scalars(epoch, epoch_iter, scalars)

                if total_steps % opts.save_latest_freq == 0:
                    print(tf_visualizer.green('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps)))
                    self.save('latest')

                if total_steps == opts.num_iter:
                    return

            if (epoch+1) % opts.save_epoch_freq == 0:
                print(tf_visualizer.green('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps)))
                self.save('latest')
                self.save(epoch+1)

def main(_):
    torch.manual_seed(0)
    trainer = ShapenetTrainer(opts)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)
