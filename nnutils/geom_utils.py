# -----------------------------------------------------------------------------------
# Code adapted from:
# https://github.com/akanazawa/cmr/blob/master/nnutils/geom_utils.py
#
# MIT License
#
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
# -----------------------------------------------------------------------------------

# Utils related to geometry like projection,,
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

import cv2
import math
import numpy as np
from ..utils import transformations


def sample_textures(texture_flow, images):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    images: B x 3 x N x N

    output: B x F x T x T x 3
    """
    # Reshape into B x F x T*T x 2
    T = texture_flow.size(-2)
    F = texture_flow.size(1)
    C = images.size(1)
    flow_grid = texture_flow.view(-1, F, T * T, 2)
    # B x 3 x F x T*T
    samples = torch.nn.functional.grid_sample(images, flow_grid)
    # B x 3 x F x T x T
    samples = samples.view(-1, C, F, T, T)
    # B x F x T x T x 3
    return samples.permute(0, 2, 3, 4, 1)

def orthographic_proj(X, cam):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    """
    quat = cam[:, -4:]
    X_rot = quat_rotate(X, quat)

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    return scale * X_rot[:, :, :2] + trans

def orthographic_proj_withz(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    Orth preserving the z.
    """
    quat = cam[:, -4:]
    X_rot = quat_rotate(X, quat)

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    proj = scale * X_rot

    proj_xy = proj[:, :, :2] + trans
    proj_z = proj[:, :, 2, None] + offset_z

    return torch.cat((proj_xy, proj_z), 2)


def cross_product(qa, qb):
    """Cross product of va by vb.

    Args:
        qa: B X N X 3 vectors
        qb: B X N X 3 vectors
    Returns:
        q_mult: B X N X 3 vectors
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]

    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]

    # See https://en.wikipedia.org/wiki/Cross_product
    q_mult_0 = qa_1*qb_2 - qa_2*qb_1
    q_mult_1 = qa_2*qb_0 - qa_0*qb_2
    q_mult_2 = qa_0*qb_1 - qa_1*qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2], dim=-1)


def hamilton_product(qa, qb):
    """Multiply qa by qb.

    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    qa_3 = qa[:, :, 3]

    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]
    qb_3 = qb[:, :, 3]

    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0*qb_0 - qa_1*qb_1 - qa_2*qb_2 - qa_3*qb_3
    q_mult_1 = qa_0*qb_1 + qa_1*qb_0 + qa_2*qb_3 - qa_3*qb_2
    q_mult_2 = qa_0*qb_2 - qa_1*qb_3 + qa_2*qb_0 + qa_3*qb_1
    q_mult_3 = qa_0*qb_3 + qa_1*qb_2 - qa_2*qb_1 + qa_3*qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)


def quat_rotate(X, q):
    """Rotate points by quaternions.

    Args:
        X: B X N X 3 points
        q: B X 4 quaternions

    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    # repeat q along 2nd dim
    ones_x = X[[0], :, :][:, :, [0]]*0 + 1
    q = torch.unsqueeze(q, 1)*ones_x

    q_conj = torch.cat([ q[:, :, [0]] , -1*q[:, :, 1:4] ], dim=-1)
    X = torch.cat([ X[:, :, [0]]*0, X ], dim=-1)

    X_rot = hamilton_product(q, hamilton_product(X, q_conj))
    return X_rot[:, :, 1:4]

def rotate_cam(cam, angle=90, axis=[0,1,0], extra_elev=False):
    """Rotate camera along some axis.
    """
    bs = cam.size(0)
    new_cams = torch.zeros(bs, 7)

    for cnt in range(bs):
        quat = cam[cnt, -4:].view(1,1,-1)

        R = transformations.quaternion_matrix(
            quat.squeeze().data.cpu().numpy())[:3, :3]
        rad_angle = np.deg2rad(angle[cnt])
        rotate_by = cv2.Rodrigues(rad_angle * np.array(axis))[0]

        new_R = rotate_by.dot(R)
        # Make homogeneous
        new_R = np.vstack(
            [np.hstack((new_R, np.zeros((3, 1)))),
             np.array([0, 0, 0, 1])])
        new_quat = transformations.quaternion_from_matrix(
            new_R, isprecise=True)
        new_quat = torch.Tensor(new_quat).cuda()
        new_cam = torch.cat([cam[cnt,:-4], new_quat], 0)
        # new_cam = torch.cat([new_ext, new_quat], 0)
        new_cams[cnt] = new_cam

    return new_cams.cuda()

def convert_ax_angle_to_quat(ax, ang):
    """
    Convert Euler angles to quaternion.
    """
    qw = torch.cos(ang/2)
    qx = ax[0] * torch.sin(ang/2)
    qy = ax[1] * torch.sin(ang/2)
    qz = ax[2] * torch.sin(ang/2)
    quat = torch.stack([qw, qx, qy, qz], dim=1)
    return quat

def ang2quat(angles):
    """
    Convert a rotation to rotation angle.

    Args:
        angles: (array): write your description
    """
    # convert from angles to quaternion
    axis = torch.eye(3).float().cuda()
    ang = torch.tanh(angles)

    azimuth = math.pi/6 * ang[...,0]
    elev = math.pi/2 * (ang[...,1])
    cyc_rot = math.pi/3 * (ang[...,2])

    q_az = convert_ax_angle_to_quat(axis[1], azimuth)
    q_el = convert_ax_angle_to_quat(axis[0], elev)
    q_cr = convert_ax_angle_to_quat(axis[2], cyc_rot)
    quat = hamilton_product(q_el.unsqueeze(1), q_az.unsqueeze(1))
    quat = hamilton_product(q_cr.unsqueeze(1), quat)
    quat = quat.squeeze(1)
    return quat
