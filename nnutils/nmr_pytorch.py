# -----------------------------------------------------------------------------------
# Code adapted from: 
# https://github.com/akanazawa/cmr/blob/master/nnutils/nmr.py
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
import tqdm

import torch
import torch.nn as nn
import neural_renderer
from ..nnutils import geom_utils

class NMR(object):
    def __init__(self, image_size, anti_aliasing, camera_mode, perspective):
        renderer = neural_renderer.Renderer(image_size=image_size, anti_aliasing=anti_aliasing, camera_mode=camera_mode, perspective=perspective, background_color=[0,0,0])
        self.renderer = renderer

    def forward_mask(self, vertices, faces):
        ''' Renders masks.
        Args:
            vertices: B X N X 3 numpy array
            faces: B X F X 3 numpy array
        Returns:
            masks: B X 256 X 256 numpy array
        '''
        masks = self.renderer.render_silhouettes(vertices, faces)
        return masks

    def forward_img(self, vertices, faces, textures):
        ''' Renders masks.
        Args:
            vertices: B X N X 3 numpy array
            faces: B X F X 3 numpy array
            textures: B X F X T X T X T X 3 numpy array
        Returns:
            images: B X 3 x 256 X 256 numpy array
        '''
        images = self.renderer.render_rgb(vertices, faces, textures)
        return images

class Render(nn.Module):
    def __init__(self, renderer):
        super(Render, self).__init__()
        self.renderer = renderer

    def forward(self, vertices, faces, textures = None):
        # B x N x 3
        # Flipping the y-axis here to make it align with the image coordinate system!
        vs = vertices
        vs[:, :, 1] *= -1
        fs = faces
        if textures is None:
            self.mask_only = True
            masks = self.renderer.forward_mask(vs, fs)
            return masks
        else:
            self.mask_only = False
            ts = textures
            imgs = self.renderer.forward_img(vs, fs, ts)
            return imgs

class NeuralRenderer(nn.Module):
    def __init__(self, img_size = 256):
        super(NeuralRenderer, self).__init__()
        self.renderer = NMR(image_size=img_size, anti_aliasing=True, camera_mode='look_at', perspective=False)

        # Set a default camera to be at (0, 0, -2.732)
        self.renderer.renderer.eye = [0, 0, -2.732]

        # Make it a bit brighter for vis
        self.renderer.renderer.light_intensity_ambient = 0.8

        self.proj_fn = geom_utils.orthographic_proj_withz
        self.offset_z = 5.

    def ambient_light_only(self):
        # Make light only ambient.
        self.renderer.renderer.light_intensity_ambient = 1
        self.renderer.renderer.light_intensity_directional = 0

    def set_bgcolor(self, color):
        self.renderer.renderer.background_color = color

    def set_light_dir(self, direction, int_dir=0.8, int_amb=0.8):
        renderer = self.renderer.renderer
        renderer.light_direction = direction
        renderer.light_intensity_directional = int_dir
        renderer.light_intensity_ambient = int_amb

    def project_points(self, verts, cams):
        proj = self.proj_fn(verts, cams)
        return proj[:, :, :2]

    def forward(self, vertices, faces, cams, textures=None):
        faces = faces.int()
        verts = self.proj_fn(vertices, cams, offset_z=self.offset_z)

        if textures is not None:
            return Render(self.renderer)(verts, faces, textures)
        else:
            return Render(self.renderer)(verts, faces)
