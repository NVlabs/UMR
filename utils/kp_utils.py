# -----------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Xueting Li.
# -----------------------------------------------------------
import torch
import torch.nn as nn

import cv2
import numpy as np
import torchvision.utils as vutils

def create_grid(F_size, GPU=True):
    b, c, h, w = F_size
    theta = torch.tensor([[1,0,0],[0,1,0]])
    theta = theta.unsqueeze(0).repeat(b,1,1)
    theta = theta.float()

    # grid is a uniform grid with left top (-1,1) and right bottom (1,1)
    # b * (h*w) * 2
    grid = nn.functional.affine_grid(theta, F_size)
    if(GPU):
    	grid = grid.cuda()
    return grid

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def draw_labelmap(img,pt,sigma):
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img)

def vis_part_heatmaps(response_maps, threshold=0.5, prefix=''):
    B,K,H,W = response_maps.shape
    part_response = np.zeros((B,K,H,W,3)).astype(np.uint8)

    for b in range(B):
        for k in range(K):
            response_map = response_maps[b,k,...].cpu().numpy()
            response_map = cv2.applyColorMap((response_map*255.0).astype(np.uint8), cv2.COLORMAP_HOT)[:,:,::-1] # BGR->RGB
            part_response[b,k,:,:,:] = response_map.astype(np.uint8)

    part_response = part_response.transpose(0,1,4,2,3)
    part_response = torch.tensor(part_response.astype(np.float32))
    results = None
    for k in range(K):
        map_viz_single = vutils.make_grid(part_response[:,k,:,:,:].squeeze()/255.0, normalize=False, scale_each=False)
        if(results is None):
            results = map_viz_single.unsqueeze(0)
        else:
            results = torch.cat((results, map_viz_single.unsqueeze(0)), dim = 0)
    return results
