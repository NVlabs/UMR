# -----------------------------------------------------------------------------
# Code adapted from:
# https://github.com/akanazawa/cmr/blob/master/data/cub.py
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
# -----------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np

import scipy.io as sio
from absl import flags, app

import torch
from torch.utils.data import Dataset

from . import base as base_data
from ..utils import transformations

# -------------- flags ------------- #
# ---------------------------------- #
flags.DEFINE_string('cub_dir', 'CUB_200_2011/', 'CUB Data Directory')
flags.DEFINE_string('cub_cache_dir', 'unsup-mesh/cachedir/cub/' , 'CUB Data Directory')
flags.DEFINE_string('scops_path', 'SCOPS/results/cub/ITER_60000/train/dcrf_prob', 'path to SCOPS results')

opts = flags.FLAGS

# -------------- Dataset ------------- #
# ------------------------------------ #
class CUBDataset(base_data.BaseDataset):
    '''
    CUB Data loader
    '''

    def __init__(self, opts, filter_key=None, mirror=True, ch=1):
        """
        Initialize annotations.

        Args:
            self: (todo): write your description
            opts: (todo): write your description
            filter_key: (str): write your description
            mirror: (str): write your description
            ch: (todo): write your description
        """
        super(CUBDataset, self).__init__(opts, filter_key=filter_key)
        self.data_dir = opts.cub_dir
        self.data_cache_dir = opts.cub_cache_dir

        self.img_dir = osp.join(self.data_dir, 'images')
        self.anno_path = osp.join(self.data_cache_dir, 'data', '%s_cub_cleaned.mat' % opts.split)
        self.anno_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % opts.split)

        if not osp.exists(self.anno_path):
            print('%s doesnt exist!' % self.anno_path)
            import ipdb; ipdb.set_trace()
        self.filter_key = filter_key

        # Load the annotation file.
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.num_imgs = len(self.anno)
        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1;
        self.mirror = mirror
        self.ch = ch


#----------- Data Loader ----------#
#----------------------------------#
def data_loader(opts, shuffle=True, mirror=True, ch=1):
    """
    Shuffle data loader.

    Args:
        opts: (todo): write your description
        shuffle: (bool): write your description
        mirror: (todo): write your description
        ch: (todo): write your description
    """
    return base_data.base_loader(CUBDataset, opts.batch_size, opts, filter_key=None,
                                 shuffle=shuffle, mirror=mirror,
                                 ch=ch)


def kp_data_loader(batch_size, opts):
    """
    Returns kp dataset.

    Args:
        batch_size: (int): write your description
        opts: (todo): write your description
    """
    return base_data.base_loader(CUBDataset, batch_size, opts, filter_key='kp')


def mask_data_loader(batch_size, opts):
    """
    Mask a dataset for a dataset.

    Args:
        batch_size: (int): write your description
        opts: (todo): write your description
    """
    return base_data.base_loader(CUBDataset, batch_size, opts, filter_key='mask')


def sfm_data_loader(batch_size, opts):
    """
    Load cifo dataset.

    Args:
        batch_size: (int): write your description
        opts: (todo): write your description
    """
    return base_data.base_loader(CUBDataset, batch_size, opts, filter_key='sfm_pose')
