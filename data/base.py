# -----------------------------------------------------------------------------
# Code adapted from https://github.com/akanazawa/cmr/blob/master/data/base.py
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

import scipy.misc
import scipy.linalg
import scipy.ndimage.interpolation
from absl import flags, app

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from ..utils import image as image_utils
from ..utils import transformations


flags.DEFINE_integer('img_size', 256, 'image size')

flags.DEFINE_float('padding_frac', 0.05,
                   'bbox is increased by this fraction of max_dim')

flags.DEFINE_float('jitter_frac', 0.05,
                   'bbox is jittered by this fraction of max_dim')

flags.DEFINE_enum('split', 'train', ['train', 'val', 'all', 'test'], 'eval split')
flags.DEFINE_integer('num_kps', 15, 'The dataloader should override these.')
flags.DEFINE_integer('n_data_workers', 4, 'Number of data loading workers')


# -------------- Dataset ------------- #
# ------------------------------------ #
class BaseDataset(Dataset):
    '''
    img, mask, kp, pose data loader
    '''

    def __init__(self, opts, filter_key=None, mirror=True, head_points_num=10,
                 belly_points_num=30, neck_points_num = 10, back_points_num = 30):
        """
        Initialize the jitter

        Args:
            self: (todo): write your description
            opts: (todo): write your description
            filter_key: (str): write your description
            mirror: (str): write your description
            head_points_num: (int): write your description
            belly_points_num: (int): write your description
            neck_points_num: (int): write your description
            back_points_num: (int): write your description
        """
        # Child class should define/load:
        # self.kp_perm
        # self.img_dir
        # self.anno
        # self.anno_sfm
        self.opts = opts
        self.img_size = opts.img_size
        self.jitter_frac = opts.jitter_frac
        self.padding_frac = opts.padding_frac
        self.filter_key = filter_key
        self.mirror = mirror
        if(opts.use_scops):
            self.scops_path = opts.scops_path

        self.head_points_num = head_points_num
        self.neck_points_num = neck_points_num
        self.belly_points_num = belly_points_num
        self.back_points_num = back_points_num

    def forward_img(self, index):
        """
        Forward image

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        data = self.anno[index]
        data_sfm = self.anno_sfm[index]

        sfm_pose = [np.copy(data_sfm.scale), np.copy(data_sfm.trans), np.copy(data_sfm.rot)]

        sfm_rot = np.pad(sfm_pose[2], (0,1), 'constant')
        sfm_rot[3, 3] = 1
        sfm_pose[2] = transformations.quaternion_from_matrix(sfm_rot, isprecise=True)

        img_path = osp.join(self.img_dir, str(data.rel_path))
        if("n02391049" in img_path.split("/")[-1]):
            # special path for zebra
            img_path = img_path.replace("JPEG", "jpg")
        img = scipy.misc.imread(img_path) / 255.0
        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        mask = np.expand_dims(data.mask, 2)

        # Adjust to 0 indexing
        bbox = np.array(
            [data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2],
            float) - 1

        parts = data.parts.T.astype(float)
        kp = np.copy(parts)
        vis = kp[:, 2] > 0
        kp[vis, :2] -= 1

        # Peturb bbox
        if self.opts.split == 'train':
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=self.jitter_frac)
        else:
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=0)
        bbox = image_utils.square_bbox(bbox)

        # crop image around bbox, translate kps
        img, mask, kp, sfm_pose = self.crop_image(img, mask, bbox, kp, vis, sfm_pose)

        # scale image, and mask. And scale kps.
        img, mask, kp, sfm_pose = self.scale_image(img, mask, kp, vis, sfm_pose)

        # read part segmentation map
        if(self.opts.use_scops):
            tmp = img_path.split("/")
            img_nm = tmp[-1].split(".")[0] + "_" + str(index) + ".npy"
            part_seg_path = osp.join(self.scops_path, tmp[-2], img_nm)
            part_prob = np.load(part_seg_path)

        # Mirror image on random.
        if self.mirror:
            if(self.opts.use_scops):
                img, mask, kp, sfm_pose, part_prob = self.mirror_image(img, mask, kp, sfm_pose, part_prob)
            else:
                img, mask, kp, sfm_pose = self.mirror_image(img, mask, kp, sfm_pose)

        # Normalize kp to be [-1, 1]
        img_h, img_w = img.shape[:2]

        kp_norm, sfm_pose = self.normalize_kp(kp, sfm_pose, img_h, img_w)

        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))

        # estimate center of the second channel, i.e. center of bird head
        if(self.opts.use_scops):
            head_center = image_utils.prob2center(part_prob, mask, ch = 1)
            neck_center = image_utils.prob2center(part_prob, mask, ch = 2)
            back_center = image_utils.prob2center(part_prob, mask, ch = 3)
            belly_center = image_utils.prob2center(part_prob, mask, ch = 4)

            # sampled points
            head_points = image_utils.sample_prob(part_prob, mask, head_center, ch = 1, num_samples = self.head_points_num)
            neck_points = image_utils.sample_prob(part_prob, mask, neck_center, ch = 2, num_samples = self.neck_points_num)
            back_points = image_utils.sample_prob(part_prob, mask, back_center, ch = 3, num_samples = self.back_points_num)
            belly_points = image_utils.sample_prob(part_prob, mask, belly_center, ch = 4, num_samples = self.belly_points_num)

        outputs = {}
        outputs['img'] = img
        outputs['kp'] = kp_norm
        outputs['sfm_pose'] = sfm_pose
        outputs['img_path'] = img_path
        outputs['mask'] = mask
        if(self.opts.use_scops):
            outputs['head_center'] = head_center
            outputs['belly_center'] = belly_center
            outputs['neck_center'] = neck_center
            outputs['back_center'] = back_center
            outputs['part_prob'] = part_prob

            outputs['head_points'] = head_points
            outputs['belly_points'] = belly_points
            outputs['neck_points'] = neck_points
            outputs['back_points'] = back_points

        return outputs

    def normalize_kp(self, kp, sfm_pose, img_h, img_w):
        """
        Normalize kp ( kp )

        Args:
            self: (todo): write your description
            kp: (todo): write your description
            sfm_pose: (bool): write your description
            img_h: (todo): write your description
            img_w: (todo): write your description
        """
        vis = kp[:, 2, None] > 0
        new_kp = np.stack([2 * (kp[:, 0] / img_w) - 1,
                           2 * (kp[:, 1] / img_h) - 1,
                           kp[:, 2]]).T
        sfm_pose[0] *= (1.0/img_w + 1.0/img_h)
        sfm_pose[1][0] = 2.0 * (sfm_pose[1][0] / img_w) - 1
        sfm_pose[1][1] = 2.0 * (sfm_pose[1][1] / img_h) - 1
        new_kp = vis * new_kp

        return new_kp, sfm_pose

    def crop_image(self, img, mask, bbox, kp, vis, sfm_pose):
        """
        Crop the image to the provided image.

        Args:
            self: (todo): write your description
            img: (array): write your description
            mask: (array): write your description
            bbox: (tuple): write your description
            kp: (todo): write your description
            vis: (todo): write your description
            sfm_pose: (todo): write your description
        """
        # crop image and mask and translate kps
        img = image_utils.crop(img, bbox, bgval=1)
        mask = image_utils.crop(mask, bbox, bgval=0)
        kp[vis, 0] -= bbox[0]
        kp[vis, 1] -= bbox[1]
        sfm_pose[1][0] -= bbox[0]
        sfm_pose[1][1] -= bbox[1]
        return img, mask, kp, sfm_pose

    def scale_image(self, img, mask, kp, vis, sfm_pose):
        """
        Scale the image

        Args:
            self: (todo): write your description
            img: (array): write your description
            mask: (array): write your description
            kp: (todo): write your description
            vis: (todo): write your description
            sfm_pose: (todo): write your description
        """
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = self.img_size / float(max(bwidth, bheight))
        img_scale, _ = image_utils.resize_img(img, scale)

        mask_scale, _ = image_utils.resize_img(mask, scale)
        kp[vis, :2] *= scale
        sfm_pose[0] *= scale
        sfm_pose[1] *= scale

        return img_scale, mask_scale, kp, sfm_pose

    def mirror_image(self, img, mask, kp, sfm_pose, part_map = None):
        """
        Mirror image

        Args:
            self: (todo): write your description
            img: (array): write your description
            mask: (array): write your description
            kp: (todo): write your description
            sfm_pose: (bool): write your description
            part_map: (str): write your description
        """
        kp_perm = self.kp_perm
        if np.random.rand(1) > 0.5:
            # Need copy bc torch collate doesnt like neg strides
            img_flip = img[:, ::-1, :].copy()
            mask_flip = mask[:, ::-1].copy()

            # Flip kps.
            new_x = img.shape[1] - kp[:, 0] - 1
            kp_flip = np.hstack((new_x[:, None], kp[:, 1:]))
            kp_flip = kp_flip[kp_perm, :]
            # Flip sfm_pose Rot.
            R = transformations.quaternion_matrix(sfm_pose[2]) # 4x4
            flip_R = np.diag([-1, 1, 1, 1]).dot(R.dot(np.diag([-1, 1, 1, 1])))
            sfm_pose[2] = transformations.quaternion_from_matrix(flip_R, isprecise=True)
            # Flip tx
            tx = img.shape[1] - sfm_pose[1][0] - 1
            sfm_pose[1][0] = tx
            if(part_map is None):
                return img_flip, mask_flip, kp_flip, sfm_pose
            else:
                part_flip = part_map[:, ::-1, :].copy()
                return img_flip, mask_flip, kp_flip, sfm_pose, part_flip
        else:
            if(part_map is None):
                return img, mask, kp, sfm_pose
            else:
                return img, mask, kp, sfm_pose, part_map

    def __len__(self):
        """
        Returns the number of images in the number.

        Args:
            self: (todo): write your description
        """
        return self.num_imgs

    def __getitem__(self, index):
        """
        Return index

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        outputs = self.forward_img(index)
        img = outputs['img']
        kp = outputs['kp']
        sfm_pose = outputs['sfm_pose']
        img_path = outputs['img_path']
        mask = outputs['mask']

        sfm_pose[0].shape = 1

        elem = {
            'img': img,
            'kp': kp,
            'mask': mask,
            'sfm_pose': np.concatenate(sfm_pose),
            'inds': index,
            'img_path': img_path,
        }

        if(self.opts.use_scops):
            elem['head_center'] = outputs['head_center']
            elem['belly_center'] = outputs['belly_center']
            elem['neck_center'] = outputs['neck_center']
            elem['back_center'] = outputs['back_center']
            elem['part_prob'] = outputs['part_prob']

            elem['head_points'] = outputs['head_points']
            elem['belly_points'] = outputs['belly_points']
            elem['neck_points'] = outputs['neck_points']
            elem['back_points'] = outputs['back_points']

        if self.filter_key is not None:
            if self.filter_key not in elem.keys():
                print('Bad filter key %s' % self.filter_key)
                import ipdb; ipdb.set_trace()
            if self.filter_key == 'sfm_pose':
                # Return both vis and sfm_pose
                vis = elem['kp'][:, 2]
                elem = {
                    'vis': vis,
                    'sfm_pose': elem['sfm_pose'],
                }
            else:
                elem = elem[self.filter_key]

        flip_img = img[:, :, ::-1].copy()
        elem['flip_img'] = flip_img

        flip_mask = mask[:, ::-1].copy()
        elem['flip_mask'] = flip_mask

        return elem

# ------------ Data Loader ----------- #
# ------------------------------------ #
def base_loader(d_set_func, batch_size, opts, filter_key=None, shuffle=True, mirror=True, ch=1):
    """
    Apply a function that load data store.

    Args:
        d_set_func: (todo): write your description
        batch_size: (int): write your description
        opts: (todo): write your description
        filter_key: (str): write your description
        shuffle: (bool): write your description
        mirror: (todo): write your description
        ch: (todo): write your description
    """
    dset = d_set_func(opts, filter_key=filter_key, mirror=mirror, ch=ch)
    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        drop_last=True)

def collate_fn(batch):
    '''Globe data collater.
    Assumes each instance is a dict.
    Applies different collation rules for each field.
    Args:
        batch: List of loaded elements via Dataset.__getitem__
    '''
    collated_batch = {'empty': True}
    if len(batch) > 0:
        for key in batch[0]:
            collated_batch[key] = default_collate([elem[key] for elem in batch])
        collated_batch['empty'] = False
    return collated_batch
