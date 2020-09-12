# -----------------------------------------------------------------------------------
# Code adapted from https://github.com/nileshkulkarni/csm/blob/master/csm/data/cub.py
#
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# -----------------------------------------------------------------------------------
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import collections

import scipy.misc
import scipy.linalg
import scipy.io as sio
import scipy.ndimage.interpolation
from absl import flags
import pickle as pkl
import torch
import multiprocessing
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import pdb
import copy
from datetime import datetime
import sys
import numpy as np
import pdb
import re
import scipy.misc
from . import base as base_data
import itertools

flags.DEFINE_string('dataset', 'cub', 'ImageNet|cub')
flags.DEFINE_integer('number_pairs', 1000, 'how many pairs of images to test on')
flags.DEFINE_string('cub_dir', 'CUB_200_2011/', 'CUB Data Directory')
flags.DEFINE_string('scops_path', 'SCOPS/results/cub/ITER_60000/train/dcrf_prob', 'path to SCOPS results')
flags.DEFINE_string('cub_cache_dir', 'unsup-mesh/cachedir/cub/' , 'CUB Data Directory')


class CubDataset(base_data.BaseDataset):

    def __init__(self, opts):
        super(CubDataset, self).__init__(opts,)
        self.data_dir = opts.cub_dir
        self.data_cache_dir = opts.cub_cache_dir
        self.opts = opts
        self.img_dir = osp.join(self.data_dir, 'images')
        self.anno_path = osp.join(self.data_cache_dir, 'data', '%s_cub_cleaned.mat' % opts.split)
        self.anno_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % opts.split)
        self.anno_train_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % 'train')
        self.jitter_frac = opts.jitter_frac
        self.padding_frac = opts.padding_frac
        self.img_size = opts.img_size
        if not osp.exists(self.anno_path):
            print('%s doesnt exist!' % self.anno_path)
            import ipdb
            ipdb.set_trace()

        # Load the annotation file.
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.kp3d = sio.loadmat(self.anno_train_sfm_path, struct_as_record=False,
                                squeeze_me=True)['S'].transpose().copy()
        self.num_imgs = len(self.anno)
        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1
        self.kp_names = ['Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye',
                         'LLeg', 'LWing', 'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat']
        return


class CubTestDataset(Dataset):

    def __init__(self, opts, filter_key):
        self.filter_key = filter_key
        sdset = CubDataset(opts)
        count = opts.number_pairs
        all_indices = [i for i in range(len(sdset))]
        rng = np.random.RandomState(len(sdset))
        pairs = zip(rng.choice(all_indices, count), rng.choice(all_indices, count))
        self.sdset = sdset
        self.tuples = list(pairs)

    def __len__(self,):
        tuples = copy.deepcopy(self.tuples)
        return len(tuples)

    def __getitem__(self, index):
        i1, i2 = self.tuples[index]
        b1 = self.sdset[i1]
        b2 = self.sdset[i2]

        if self.filter_key==1:
            return b1
        else:
            return b2



def cub_dataloader(opts, shuffle=True):
    dset = CubDataset(opts)
    return DataLoader(
        dset,
        batch_size=opts.batch_size,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        pin_memory=True, collate_fn=base_data.collate_fn)


def cub_test_pair_dataloader(opts, filter_key, shuffle=False):
    dset = CubTestDataset(opts, filter_key)
    return DataLoader(
        dset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        pin_memory=True, collate_fn=base_data.collate_fn)


def cub_dataset(opts):
    dset = CubDataset(opts)

    class DataIter():
        def __init__(self, dset, collate_fn):
            self.collate_fn = collate_fn
            self.dset = dset
            return

        def __len__(self,):
            return len(self.dset)

        def __getitem__(self, index):
            example = dset[index]
            return self.collate_fn([example])

    return DataIter(dset, base_data.collate_fn)
