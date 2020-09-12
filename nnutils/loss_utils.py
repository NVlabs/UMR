# -----------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Xueting Li.
# -----------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import scipy.misc
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torchvision.utils as vutils

from . import geom_utils
from . import scops_utils
from .smr import SoftRenderer
from .nmr_pytorch import NeuralRenderer
from .chamfer_python import distChamfer
from ..utils import image as image_utils

class edge_regularization(nn.Module):
    def __init__(self, edges):
        super(edge_regularization,self).__init__()
        self.edges = edges.long()

    def forward(self, pred):
        """
        :param pred: batch_size * num_points * 3
        :param edges: num_edges * 2
        :return:
        """
        l2_loss = nn.MSELoss(reduction='mean')
        return l2_loss(pred[:, self.edges[:, 0]], pred[:, self.edges[:, 1]]) * pred.size(-1)

def neg_iou_loss(predict, target, avg = True):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    if(avg):
        return 1. - (intersect / union).sum() / intersect.nelement()
    else:
        return 1. - (intersect / union)

def texture_dt_loss(texture_flow, dist_transf, vis_rend=None, cams=None, verts=None, tex_pred=None):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    dist_transf: B x 1 x N x N

    Similar to geom_utils.sample_textures
    But instead of sampling image, it samples dt values.
    """
    # Reshape into B x F x T*T x 2
    T = texture_flow.size(-2)
    F = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, F, T * T, 2)
    # B x 1 x F x T*T
    dist_transf = torch.nn.functional.grid_sample(dist_transf, flow_grid)

    if vis_rend is not None:
        # Visualize the error!
        # B x 3 x F x T*T
        dts = dist_transf.repeat(1, 3, 1, 1)
        # B x 3 x F x T x T
        dts = dts.view(-1, 3, F, T, T)
        # B x F x T x T x 3
        dts = dts.permute(0, 2, 3, 4, 1)
        dts = dts.unsqueeze(4).repeat(1, 1, 1, 1, T, 1) / dts.max()

        from ..utils import bird_vis
        for i in range(dist_transf.size(0)):
            rend_dt = vis_rend(verts[i], cams[i], dts[i])
            rend_img = bird_vis.tensor2im(tex_pred[i].data)
            import matplotlib.pyplot as plt
            plt.ion()
            fig=plt.figure(1)
            plt.clf()
            ax = fig.add_subplot(121)
            ax.imshow(rend_dt)
            ax = fig.add_subplot(122)
            ax.imshow(rend_img)
            import ipdb; ipdb.set_trace()

    return dist_transf.mean()


def texture_loss(img_pred, img_gt, mask_gt):
    """
    Input:
      img_pred, img_gt: B x 3 x H x W
      mask_pred, mask_gt: B x H x W
    """
    mask_gt = mask_gt.unsqueeze(1)

    return torch.nn.L1Loss()(img_pred * mask_gt, img_gt * mask_gt)

def texture_loss_masks(img_pred, img_gt, mask_gt, mask_pred, avg=True):
    """
    Input:
      img_pred, img_gt: B x 3 x H x W
      mask_pred, mask_gt: B x H x W
    """
    mask_gt = mask_gt.unsqueeze(1)
    mask_pred = mask_pred.unsqueeze(1)
    if(avg):
        return torch.nn.L1Loss()(img_pred * mask_pred, img_gt * mask_gt)
    else:
        loss = torch.nn.L1Loss(reduction = 'none')(img_pred * mask_pred, img_gt * mask_gt)
        loss = torch.sum(loss, dim = (1, 2, 3)) / (loss.size(1) * loss.size(2) * loss.size(3))
        return loss

def deform_l2reg(V):
    """
    l2 norm on V = B x N x 3
    """
    V = V.view(-1, V.size(2))
    return torch.mean(torch.norm(V, p=2, dim=1))

def sym_reg(verts):
    return torch.mean(torch.abs(verts[:,:,1]))

class PerceptualTextureLoss(object):
    def __init__(self):
        from .perceptual_loss import PerceptualLoss
        self.perceptual_loss = PerceptualLoss()

    def __call__(self, img_pred, img_gt, mask_gt, mask_pred=None, avg = True):
        """
        Input:
          img_pred, img_gt: B x 3 x H x W
        mask_pred, mask_gt: B x H x W
        """
        mask_gt = mask_gt.unsqueeze(1)
        if(mask_pred is not None):
            mask_pred = mask_pred.unsqueeze(1)
            dist = self.perceptual_loss(img_pred * mask_pred, img_gt * mask_gt)
        else:
            dist = self.perceptual_loss(img_pred * mask_gt, img_gt * mask_gt)

        # Only use mask_gt..
        if(avg):
            return dist.mean()
        else:
            return dist

class TexCycle(nn.Module):
    def __init__(self, im_size = 256, nf = 1280, eps = 1e-12):
        super(TexCycle,self).__init__()

    def forward(self, flow, prob, aggr_info):
        """
        INPUTS:
         - flow: learned texture flow (nb * nf * nr * nr * 2)
         - prob: affinity between image & mesh by renderer (nb * nf * 2)
         - aggr_info: provide information about visible faces.
        OUTPUTS:
         - texture cycle loss
        IDEA:
         - make averaged coords of projected face equals to predicted flow
        """
        nb, nf, nr, _, _ = flow.size()

        flow_grid = flow.view(nb, nf, -1, 2)
        avg_flow = torch.mean(flow_grid, dim = 2)

        # mask: nb x nf x 2
        # only rows correspond to visible faces are set to 1
        mask = torch.zeros(avg_flow.size())
        for cnt in range(nb):
            fids = torch.unique(aggr_info[cnt]).long()
            mask[cnt, fids, :] = 1

        mask = mask.cuda()
        loss = torch.nn.MSELoss()(avg_flow * mask, prob * mask)
        # second term for visilization purpose
        return loss, avg_flow[0, 0:10, :]

def entropy_loss(A):
    """
    Input is K x N
    Each column is a prob of vertices being the one for k-th keypoint.
    We want this to be sparse = low entropy.
    """
    entropy = -torch.sum(A * torch.log(A), 1)
    # Return avg entropy over
    return torch.mean(entropy)

class CorrLossChamfer(nn.Module):
    def __init__(self, scops_path, image_size,
                 weights = torch.Tensor([0.45, 0.45, 0.05, 0.05])):
        super(CorrLossChamfer,self).__init__()
        """
        Caculate Chamfer distance of four parts.
        The indices of each part (head_idx etc.) could be a path or torch Tensor.
        """
        head_vertices = np.load(osp.join(scops_path, "vertices_idx/head_vertices.npy"))
        self.head_vertices = torch.from_numpy(head_vertices).long()

        belly_vertices = np.load(osp.join(scops_path, "vertices_idx/belly_vertices.npy"))
        self.belly_vertices = torch.from_numpy(belly_vertices).long()

        neck_vertices = np.load(osp.join(scops_path, "vertices_idx/neck_vertices.npy"))
        self.neck_vertices = torch.from_numpy(neck_vertices).long()

        back_vertices = np.load(osp.join(scops_path, "vertices_idx/back_vertices.npy"))
        self.back_vertices = torch.from_numpy(back_vertices).long()

        self.renderer = SoftRenderer(image_size)

        # some parts with large deformation should be down-weighted
        # calculate the weights based on number of vertices in each part
        self.head_num = len(self.head_vertices)
        self.belly_num = len(self.belly_vertices)
        self.neck_num = len(self.neck_vertices)
        self.back_num = len(self.back_vertices)

        self.weights = weights

    def forward(self, head_points, belly_points, neck_points, back_points, verts, cams, avg = True):
        bs = head_points.size(0)

        head_vert_coords = verts[:, self.head_vertices, :]
        belly_vert_coords = verts[:, self.belly_vertices, :]
        back_vert_coords = verts[:, self.back_vertices, :]
        neck_vert_coords = verts[:, self.neck_vertices, :]
        vert_coords = torch.cat((head_vert_coords, belly_vert_coords, neck_vert_coords, back_vert_coords), dim = 1)

        vert2d = self.renderer.project_points(vert_coords, cams)
        cdist = None
        if(self.head_num > 0):
            head_cdist1, _, _, _ = distChamfer(vert2d[:, :self.head_num, :], head_points)
            cdist = head_cdist1 * self.weights[0]
        if(self.belly_num > 0):
            belly_cdist1, _, _, _ = distChamfer(vert2d[:, self.head_num:self.head_num + self.belly_num, :], belly_points)
            cdist = torch.cat((cdist, belly_cdist1 * self.weights[1]), dim = 1)
        if(self.neck_num > 0):
            neck_cdist1, _, _, _ = distChamfer(vert2d[:, self.head_num + self.belly_num:self.head_num + self.belly_num + self.neck_num, :], neck_points)
            cdist = torch.cat((cdist, neck_cdist1 * self.weights[2]), dim = 1)
        if(self.back_num > 0):
            back_cdist1, _, _, _ = distChamfer(vert2d[:, self.head_num + self.belly_num + self.neck_num:, :], back_points)
            cdist = torch.cat((cdist, back_cdist1 * self.weights[3]), dim = 1)

        loss = torch.mean(cdist, dim = 1)
        if(avg):
            return torch.mean(loss), vert2d
        else:
            return loss

class MultiMaskLoss(nn.Module):
    def __init__(self, image_size = 256, renderer_type = "softmax", num_hypo_cams = 8):
        super(MultiMaskLoss, self).__init__()
        self.renderer = SoftRenderer(image_size, renderer_type)
        self.num_hypo_cams = num_hypo_cams
        self.image_size = image_size

    def forward(self, vs, fs, cams_all_hypo, cam_probs, masks_gt):
        bs = vs.size(0)
        # prepare vertices, faces, cameras
        pred_vs = vs.unsqueeze(1).repeat(1, self.num_hypo_cams, 1, 1).view(-1, vs.size(1), 3)
        faces = fs.unsqueeze(1).repeat(1, self.num_hypo_cams, 1, 1).view(-1, fs.size(1), 3)
        cams_all_hypo_flat = cams_all_hypo.view(-1, 7)

        # prepare images
        pred, _, _ = self.renderer.forward(pred_vs, faces, cams_all_hypo_flat)
        mask_all_hypo = pred[:, 3, :, :]
        masks = masks_gt.unsqueeze(1).repeat(1, self.num_hypo_cams, 1, 1).view(-1, self.image_size, self.image_size)

        # calculate loss
        loss = neg_iou_loss(mask_all_hypo, masks, avg = False)
        loss = loss.view(bs, -1) * cam_probs
        loss = loss.sum(dim = 1)
        mask_loss = loss.mean()

        return mask_loss, mask_all_hypo

class MultiTextureLoss(nn.Module):
    def __init__(self, samples_per_gpu = 32, num_hypo_cams = 8,
                 image_size = 256, renderer_type = "softmax", texture_loss_type="perceptual",
                 renderer = "smr"):
        super(MultiTextureLoss, self).__init__()
        if(renderer in "smr"):
            self.renderer = SoftRenderer(image_size, renderer_type)
        else:
            self.renderer = NeuralRenderer(image_size)
        self.renderer.ambient_light_only()
        self.hard_renderer = SoftRenderer(image_size, "hard")

        if(texture_loss_type in "perceptual"):
            self.texture_loss = PerceptualTextureLoss()
        else:
            self.texture_loss = texture_loss_masks
        self.texture_cycle_fn = TexCycle(samples_per_gpu)

        self.num_hypo_cams = num_hypo_cams
        self.image_size = image_size
        self.which_renderer = renderer


    def forward(self, vs, fs, cams_all_hypo, cam_probs, proj_cam, rgbs, masks_gt, masks_pred, tx, tex_flow, dts_barrier):
        bs = vs.size(0)
        # prepare vertices, faces, textures and cameras
        pred_vs = vs.unsqueeze(1).repeat(1, self.num_hypo_cams, 1, 1).view(-1, vs.size(1), 3)
        faces = fs.unsqueeze(1).repeat(1, self.num_hypo_cams, 1, 1).view(-1, fs.size(1), 3)
        tex = tx.unsqueeze(1).repeat(1, self.num_hypo_cams, 1, 1, 1).view(-1, tx.size(1), tx.size(2), 3)
        cams_all_hypo_flat = cams_all_hypo.view(-1, 7)

        # prepare images: rendering and GT
        if(self.which_renderer in "nmr"):
            tex = tex.view(tex.size(0), tex.size(1), 6, 6, 3).unsqueeze(2).repeat(1, 1, 6, 1, 1, 1)
            texture_rgba = self.renderer.forward(pred_vs.detach(), faces, cams_all_hypo_flat, tex)
        else:
            texture_rgba, _, _ = self.renderer.forward(pred_vs.detach(), faces, cams_all_hypo_flat, tex)
        texture_pred = texture_rgba[:,0:3,:,:]
        imgs = rgbs.unsqueeze(1).repeat(1, self.num_hypo_cams, 1, 1, 1).view(-1, 3, self.image_size, self.image_size)
        masks_gt = masks_gt.unsqueeze(1).repeat(1, self.num_hypo_cams, 1, 1).view(-1, self.image_size, self.image_size)

        # calculate perceptual loss
        tex_loss = self.texture_loss(texture_pred, imgs, masks_gt, masks_pred, avg = False)
        tex_loss = tex_loss.view(bs, -1)
        tex_loss = tex_loss.to(cam_probs.get_device())
        tex_loss = (tex_loss * cam_probs).sum(dim = 1)
        tex_loss = tex_loss.mean()
        tex_dt_loss = texture_dt_loss(tex_flow, dts_barrier)

        # get the visiblility map
        _, p2f_info, aggr_info = self.hard_renderer(vs.detach(), fs, proj_cam.detach())
        aggr_info = aggr_info[:, 1, :, :].view(bs, -1)
        tex_cycle_loss, avg_flow = self.texture_cycle_fn(tex_flow, p2f_info.detach(), aggr_info.detach())

        return tex_loss, tex_dt_loss, tex_cycle_loss, texture_pred

class part_matching_loss(nn.Module):
    def __init__(self, scops_path, uv_sampler, num_sym_faces,
                 im_size = 256, batch_size = 32,
                 loss_type = 'mse', tex_size=6,
                 num_cam = 1):
        super(part_matching_loss, self).__init__()

        # load mean semantic uv map
        uv_img = scipy.misc.imread(osp.join(scops_path, "semantic_seg.png"))
        uv_img = torch.from_numpy(uv_img).view(1, 1, 128, 256).float()
        uv_img = uv_img.cuda()

        tex = torch.nn.functional.grid_sample(uv_img, uv_sampler)
        tex = tex.view(tex.size(0), -1, tex.size(2), tex_size, tex_size).permute(0, 2, 3, 4, 1)
        tex_left = tex[:, -num_sym_faces:]
        tex = torch.cat([tex, tex_left], 1)
        tex = tex.view(tex.size(0), tex.size(1), -1, 1).squeeze()
        stex = torch.round(tex)

        # to one-hot
        nf, nt = stex.size()
        one_hot = torch.zeros(nf * nt, 5).cuda()
        one_hot.scatter_(1, stex.view(-1, 1).long(), 1)
        stex_one_hot = one_hot.view(1, nf, nt, 5)

        # semantic texture for each part, because the renderer can only render
        # 3-channel images, so we renderer each part separately
        self.register_buffer("stex1", stex_one_hot[:, :, :, 1].unsqueeze(-1).repeat(batch_size * num_cam, 1, 1, 3))
        self.register_buffer("stex2", stex_one_hot[:, :, :, 2].unsqueeze(-1).repeat(batch_size * num_cam, 1, 1, 3))
        self.register_buffer("stex3", stex_one_hot[:, :, :, 3].unsqueeze(-1).repeat(batch_size * num_cam, 1, 1, 3))
        self.register_buffer("stex4", stex_one_hot[:, :, :, 4].unsqueeze(-1).repeat(batch_size * num_cam, 1, 1, 3))

        self.renderer = SoftRenderer(im_size, "softmax")
        self.renderer.ambient_light_only()
        self.kl = nn.KLDivLoss(reduction='batchmean')

        # default value for backgrounds
        self.register_buffer("proj", torch.zeros(batch_size * num_cam, 1, 256, 256))
        self.proj[:, 0, :, :] = 0.1

        # weights for each parts
        weights = torch.Tensor([0, 5.0, 0.0, 0.0, 5.0]).view(1, 5, 1, 1).cuda()
        self.register_buffer("weights", weights)

        self.loss_type = loss_type

    def forward(self, verts, faces, cams, part_segs, cam_probs = None, avg = True):
        total_loss = 0
        projs = []
        bs = verts.size(0)

        # project each part
        proj1, _, _ = self.renderer(verts, faces, cams, self.stex1[:bs])
        proj1 = torch.mean(proj1[:, 0:3, :, :], dim = 1).unsqueeze(1)
        projs.append(proj1)

        proj2, _, _ = self.renderer(verts, faces, cams, self.stex2[:bs])
        proj2 = torch.mean(proj2[:, 0:3, :, :], dim = 1).unsqueeze(1)
        projs.append(proj2)

        proj3, _, _ = self.renderer(verts, faces, cams, self.stex3[:bs])
        proj3 = torch.mean(proj3[:, 0:3, :, :], dim = 1).unsqueeze(1)
        projs.append(proj3)

        proj4, _, _ = self.renderer(verts, faces, cams, self.stex4[:bs])
        proj4 = torch.mean(proj4[:, 0:3, :, :], dim = 1).unsqueeze(1)
        projs.append(proj4)

        proj = torch.cat((self.proj[:bs].detach(), proj1, proj2, proj3, proj4), dim = 1)
        centers_proj = scops_utils.batch_get_centers(nn.Softmax(dim = 1)(proj)[:, 1:, :, :])
        centers_parts = scops_utils.batch_get_centers(nn.Softmax(dim = 1)(part_segs)[:, 1:, :, :])

        if(avg):
            loss_lmeqv = torch.nn.functional.mse_loss(centers_proj, centers_parts)
        else:
            loss_lmeqv = torch.nn.functional.mse_loss(centers_proj, centers_parts, reduction = 'none')
            loss_lmeqv = torch.sum(loss_lmeqv, dim = (1, 2)) / (loss_lmeqv.size(1) * loss_lmeqv.size(2))
            loss_lmeqv = loss_lmeqv.view(cam_probs.size())
            loss_lmeqv = loss_lmeqv.to(cam_probs.get_device())
            loss_lmeqv = (loss_lmeqv * cam_probs).sum(dim = 1)
            loss_lmeqv = loss_lmeqv.mean()

        if(self.loss_type in 'kld'):
            loss_eqv = self.kl(torch.nn.functional.log_softmax(proj, dim = 1), torch.nn.functional.softmax(part_segs, dim = 1))
        else:
            # normalize each part
            # if scale is really small, the division might cause Nan.
            max_proj, _ = torch.max(proj.view(bs, 5, -1), dim = 2)
            max_proj[max_proj < 1e-5] = 1e-5
            proj_norm = proj / max_proj.view(bs, 5, 1, 1)

            max_part, _ = torch.max(part_segs.view(bs, 5, -1), dim = 2)
            max_part[max_part < 1e-5] = 1e-5
            part_norm = part_segs / max_part.view(bs, 5, 1, 1)

            if(avg):
                loss_eqv = torch.mean(nn.MSELoss(reduction='none')(proj_norm, part_norm) * self.weights)
            else:
                bs, cs, iis, iis = part_norm.size()
                loss_eqv = nn.MSELoss(reduction='none')(proj_norm, part_norm) * self.weights
                loss_eqv = torch.sum(loss_eqv, dim = (1, 2, 3)) / (cs * iis * iis)
                loss_eqv = loss_eqv.view(cam_probs.size())
                loss_eqv = loss_eqv.to(cam_probs.get_device())
                loss_eqv = (loss_eqv * cam_probs).sum(dim = 1)
                loss_eqv = loss_eqv.mean()

        total_loss = loss_eqv + loss_lmeqv
        return total_loss / 4.0, projs
