# -----------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Xueting Li.
# -----------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import copy
import numpy as np
import os.path as osp
from absl import app, flags

import torch
import torchvision
import torch.nn as nn

from ..utils import mesh
from . import net_blocks as nb

#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_boolean('symmetric', True, 'Use symmetric mesh or not')
flags.DEFINE_integer('nz_feat', 200, 'Encoded image feature size')
flags.DEFINE_integer('z_dim', 350, 'noise dimension of VAE')
flags.DEFINE_integer('gpu_num', 1, 'gpu number')

flags.DEFINE_boolean('use_texture', True, 'if true uses texture!')
flags.DEFINE_boolean('symmetric_texture', True, 'if true texture is symmetric!')
flags.DEFINE_integer('tex_size', 6, 'Texture resolution per face')

flags.DEFINE_integer('subdivide', 3, '# to subdivide icosahedron, 3=642verts, 4=2562 verts')

flags.DEFINE_boolean('use_deconv', False, 'If true uses Deconv')
flags.DEFINE_string('upconv_mode', 'bilinear', 'upsample mode')

flags.DEFINE_boolean('only_mean_sym', False, 'If true, only the meanshape is symmetric')


#------------- Modules ------------#
#----------------------------------#
class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x


class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, n_blocks=4, nz_feat=100, batch_norm=True, z_dim=200):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(n_blocks=4)
        self.enc_conv1 = nb.conv2d(batch_norm, 512, 256, stride=2, kernel_size=4)
        nc_input = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)
        self.enc_fc = nb.fc_stack(nc_input, nz_feat, 2)

        self.mean_fc = nn.Sequential(nn.Linear(nz_feat, nz_feat),
                                     nn.LeakyReLU(),
                                     nn.Linear(nz_feat, z_dim))

        self.logvar_fc = nn.Sequential(nn.Linear(nz_feat, nz_feat),
                                       nn.LeakyReLU(),
                                       nn.Linear(nz_feat, z_dim))

        nb.net_init(self.enc_conv1)

    def sampling(self, mu, logvar):
        var = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()
        eps = eps.cuda()
        return eps.mul(var).add_(mu)

    def forward(self, img):
        resnet_feat = self.resnet_conv(img)

        out_enc_conv1 = self.enc_conv1(resnet_feat)
        out_enc_conv1 = out_enc_conv1.view(img.size(0), -1)
        feat = self.enc_fc(out_enc_conv1)

        mean = self.mean_fc(feat)
        logvar = self.logvar_fc(feat)

        return feat, self.sampling(mean, logvar), mean, logvar

class TexturePredictorUV(nn.Module):
    """
    Outputs mesh texture
    """

    def __init__(self, nz_feat, F, T, opts, img_H=64, img_W=128, n_upconv=5, nc_init=256, predict_flow=False, symmetric=False, num_sym_faces=624):
        super(TexturePredictorUV, self).__init__()
        self.feat_H = img_H // (2 ** n_upconv)
        self.feat_W = img_W // (2 ** n_upconv)
        self.nc_init = nc_init
        self.symmetric = symmetric
        self.num_sym_faces = num_sym_faces
        self.F = F
        self.T = T
        self.predict_flow = predict_flow

        self.enc = nb.fc_stack(nz_feat, self.nc_init*self.feat_H*self.feat_W, 2)
        if predict_flow:
            nc_final=2
        else:
            nc_final=3
        self.decoder = nb.decoder2d(n_upconv, None, nc_init, init_fc=False, nc_final=nc_final, use_deconv=opts.use_deconv, upconv_mode=opts.upconv_mode)

    def forward(self, feat, uv_sampler):
        # pdb.set_trace()
        bs = feat.size(0)
        uvimage_pred = self.enc(feat)
        uvimage_pred = uvimage_pred.view(uvimage_pred.size(0), self.nc_init, self.feat_H, self.feat_W)
        # B x 2 or 3 x H x W
        self.uvimage_pred = self.decoder(uvimage_pred)
        self.uvimage_pred = torch.nn.functional.tanh(self.uvimage_pred)

        tex_pred = torch.nn.functional.grid_sample(self.uvimage_pred, uv_sampler)
        tex_pred = tex_pred.view(tex_pred.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1)

        if self.symmetric:
            # Symmetrize.
            tex_left = tex_pred[:, -self.num_sym_faces:]
            return torch.cat([tex_pred, tex_left], 1), self.uvimage_pred
        else:
            # Contiguous Needed after the permute..
            return tex_pred.contiguous(), self.uvimage_pred



class ShapePredictor(nn.Module):
    """
    Outputs mesh deformations
    """

    def __init__(self, nz_feat, num_verts):
        super(ShapePredictor, self).__init__()
        self.pred_layer = nn.Sequential(nn.Linear(nz_feat, nz_feat),
                                        nn.LeakyReLU(True),
                                        nn.Linear(nz_feat, nz_feat*2),
                                        nn.LeakyReLU(True),
                                        nn.Linear(nz_feat*2, nz_feat*4),
                                        nn.LeakyReLU(True),
                                        nn.Linear(nz_feat*4, nz_feat*8),
                                        nn.LeakyReLU(True),
                                        nn.Linear(nz_feat*8, num_verts*3),
                                        )

    def forward(self, feat):
        delta_v = self.pred_layer(feat)
        # Make it B x num_verts x 3
        delta_v = delta_v.view(delta_v.size(0), -1, 3)

        return delta_v


class QuatPredictor(nn.Module):
    def __init__(self, nz_feat, nz_rot=4, classify_rot=False):
        super(QuatPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, nz_rot)
        self.classify_rot = classify_rot

    def forward(self, feat):
        quat = self.pred_layer(feat)
        if self.classify_rot:
            quat = torch.nn.functional.log_softmax(quat)
        else:
            quat = torch.nn.functional.normalize(quat)
        return quat


class ScalePredictor(nn.Module):
    def __init__(self, nz):
        super(ScalePredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 1)

    def forward(self, feat):
        scale = self.pred_layer(feat) + 1  #biasing the scale to 1
        scale = torch.nn.functional.relu(scale) + 1e-12
        return scale


class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """

    def __init__(self, nz, orth=True):
        super(TransPredictor, self).__init__()
        if orth:
            self.pred_layer = nn.Linear(nz, 2)
        else:
            self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        trans = self.pred_layer(feat)
        return trans


#------------ Mesh Net ------------#
#----------------------------------#
class MeshNet(nn.Module):
    def __init__(self, input_shape, opts, nz_feat = 100, axis = 0,
                 temp_dir = None):
        # Input shape is H x W of the image.
        super(MeshNet, self).__init__()
        self.opts = opts
        self.pred_texture = opts.use_texture
        self.symmetric = opts.symmetric
        self.symmetric_texture = opts.symmetric_texture
        self.pred_cam = opts.pred_cam
        self.nz_feat = nz_feat
        self.z_dim = opts.z_dim
        self.batch_size = opts.batch_size

        verts, faces = mesh.create_sphere(opts.subdivide)
        num_verts = verts.shape[0]

        if self.symmetric:
            verts, faces, num_indept, num_sym, num_indept_faces, num_sym_faces = mesh.make_symmetric(verts, faces, axis=axis)

            num_sym_output = num_indept + num_sym
            self.num_output = num_sym_output
            self.num_sym = num_sym
            self.num_indept = num_indept
            self.num_indept_faces = num_indept_faces
            self.num_sym_faces = num_sym_faces

            # mean shape is only half.
            mean_v = torch.Tensor(verts[:num_sym_output])
            if(temp_dir is not None):
                mean_v = torch.load(temp_dir)
            self.register_buffer('mean_v', mean_v)

            # Needed for symmetrizing..
            self.flip = torch.ones(1, 3).cuda()
            self.flip[0, axis] = -1
        else:
            self.mean_v = nn.Parameter(torch.Tensor(verts), requires_grad=False)
            self.num_output = num_verts

        verts_np = verts
        faces_np = faces
        self.verts_np = verts_np
        self.faces_np = faces_np
        self.faces = torch.LongTensor(faces).cuda()

        self.encoder = Encoder(input_shape, n_blocks=4, nz_feat=nz_feat, z_dim=opts.z_dim)
        self.shape_predictor = ShapePredictor(opts.z_dim, num_verts=self.num_output)

        if(self.pred_cam):
            self.quat_predictor = QuatPredictor(nz_feat)
            self.scale_predictor = ScalePredictor(nz_feat)
            self.trans_predictor = TransPredictor(nz_feat)

        if self.pred_texture:
            if self.symmetric_texture:
                num_faces = self.num_indept_faces + self.num_sym_faces
            else:
                num_faces = faces.shape[0]

            uv_sampler = mesh.compute_uvsampler(verts_np, faces_np[:num_faces], tex_size=opts.tex_size)
            uv_sampler = torch.FloatTensor(uv_sampler).cuda()
            uv_sampler = uv_sampler.unsqueeze(0).repeat(int(self.opts.batch_size/self.opts.gpu_num), 1, 1, 1, 1)
            self.F = uv_sampler.size(1)
            self.T = uv_sampler.size(2)

            uv_sampler = uv_sampler.view(-1, self.F, self.T*self.T, 2)
            self.register_buffer('uv_sampler', uv_sampler)

            img_H = int(2**np.floor(np.log2(np.sqrt(num_faces) * opts.tex_size)))
            img_W = 2 * img_H
            self.num_faces = num_faces
            if(self.symmetric_texture):
                self.texture_predictor = TexturePredictorUV(
                  nz_feat, self.F, self.T, opts, img_H=img_H, img_W=img_W, predict_flow=True, symmetric=opts.symmetric_texture, num_sym_faces=self.num_sym_faces)
            else:
                self.texture_predictor = TexturePredictorUV(
                  nz_feat, self.F, self.T, opts, img_H=img_H, img_W=img_W, predict_flow=True, symmetric=opts.symmetric_texture)
            nb.net_init(self.texture_predictor)
            self.tex_size = opts.tex_size
        self.register_buffer('noise', torch.zeros(int(self.batch_size/self.opts.gpu_num), self.z_dim))

    def forward(self, img=None):
        outputs = {}
        # reconstruct path
        img_feat, noise, mean, logvar = self.encoder(img)
        shape_pred = self.shape_predictor(noise)
        if(self.pred_cam):
            scale_pred = self.scale_predictor(img_feat)
            quat_pred = self.quat_predictor(img_feat)
            trans_pred = self.trans_predictor(img_feat)
            cam_pred = torch.cat((scale_pred, trans_pred, quat_pred), dim = 1)
            outputs['cam'] = cam_pred

        if self.pred_texture:
            if(self.uv_sampler.size(0) != img_feat.size(0)):
                uv_sampler = self.uv_sampler[0].unsqueeze(0).repeat(img_feat.size(0), 1, 1, 1)
                texture_pred, uvimage_pred = self.texture_predictor(img_feat, uv_sampler)
            else:
                texture_pred, uvimage_pred = self.texture_predictor(img_feat, self.uv_sampler)
            outputs['tex_flow'] = texture_pred
            outputs['uvimage_pred'] = uvimage_pred
        outputs['feat'] = noise

        outputs['delta_v'] = shape_pred

        return outputs

    def symmetrize(self, V):
        """
        Takes num_indept+num_sym verts and makes it
        num_indept + num_sym + num_sym
        Is identity if model is not symmetric
        """
        if self.symmetric:
            if V.dim() == 2:
                # No batch
                V_left = self.flip * V[-self.num_sym:]
                return torch.cat([V, V_left], 0)
            else:
                # With batch
                V_left = self.flip * V[:, -self.num_sym:]
                return torch.cat([V, V_left], 1)
        else:
            return V

    def get_mean_shape(self):
        return self.symmetrize(self.mean_v)
