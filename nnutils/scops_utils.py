# -----------------------------------------------------------------------------------
# Code adapted from:
# https://github.com/NVlabs/SCOPS/blob/master/utils/utils.py
#
# Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
# -----------------------------------------------------------------------------------

import numpy as np
import torch

def get_coordinate_tensors(x_max, y_max):
    """
    Get the coordinates of the coordinates of a tensorors.

    Args:
        x_max: (int): write your description
        y_max: (todo): write your description
    """
    x_map = np.tile(np.arange(x_max), (y_max,1))/x_max*2 - 1.0
    y_map = np.tile(np.arange(y_max), (x_max,1)).T/y_max*2 - 1.0

    x_map_tensor = torch.from_numpy(x_map.astype(np.float32)).cuda()
    y_map_tensor = torch.from_numpy(y_map.astype(np.float32)).cuda()

    return x_map_tensor, y_map_tensor

def get_center(part_map, self_referenced=False):
    """
    Return the center of the map.

    Args:
        part_map: (str): write your description
        self_referenced: (str): write your description
    """

    h,w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h,w)

    x_center = (part_map * x_map).sum()
    y_center = (part_map * y_map).sum()

    if self_referenced:
        x_c_value = float(x_center.cpu().detach())
        y_c_value = float(y_center.cpu().detach())
        x_center = (part_map * (x_map - x_c_value)).sum() + x_c_value
        y_center = (part_map * (y_map - y_c_value)).sum() + y_c_value

    return x_center, y_center

def get_centers(part_maps, detach_k=True, epsilon=1e-3, self_ref_coord=False):
    """
    Return the coordinates of the coordinates of the centers.

    Args:
        part_maps: (str): write your description
        detach_k: (str): write your description
        epsilon: (float): write your description
        self_ref_coord: (str): write your description
    """
    C,H,W = part_maps.shape
    centers = []
    for c in range(C):
        part_map = part_maps[c,:,:] + epsilon
        k = part_map.sum()
        part_map_pdf = part_map/k
        x_c, y_c = get_center(part_map_pdf, self_ref_coord)
        centers.append(torch.stack((x_c, y_c), dim=0).unsqueeze(0))
    return torch.cat(centers, dim=0)

def batch_get_centers(pred_softmax):
    """
    Get the number of the tensors.

    Args:
        pred_softmax: (int): write your description
    """
    B,C,H,W = pred_softmax.shape

    centers_list = []
    for b in range(B):
        centers_list.append(get_centers(pred_softmax[b]).unsqueeze(0))
    return torch.cat(centers_list, dim=0)

class Colorize(object):
    def __init__(self, n=22):
        """
        Initialize the color map.

        Args:
            self: (todo): write your description
            n: (int): write your description
        """
        self.cmap = color_map(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        """
        Call the color image.

        Args:
            self: (todo): write your description
            gray_image: (array): write your description
        """
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

def color_map(N=256, normalized=False):
    """
    Return a color map.

    Args:
        N: (int): write your description
        normalized: (bool): write your description
    """
    def bitget(byteval, idx):
        """
        Return the value of a byteval.

        Args:
            byteval: (todo): write your description
            idx: (int): write your description
        """
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def denseCRF(img, pred):
    """
    Dense - f2dense )

    Args:
        img: (array): write your description
        pred: (todo): write your description
    """
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    N,H,W = pred.shape

    d = dcrf.DenseCRF2D(W, H, N)  # width, height, nlabels
    U = unary_from_softmax(pred)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=5)

    Q = d.inference(5)
    Q = np.array(Q).reshape((N,H,W)).transpose(1,2,0)

    return Q
