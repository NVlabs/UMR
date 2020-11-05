# -----------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Xueting Li.
# -----------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
import tqdm

import torch

import soft_renderer as sr

from ..nnutils import geom_utils

#############
### Utils ###
#############
def convert_as(src, trg):
    """
    Convert a device ascii.

    Args:
        src: (todo): write your description
        trg: (todo): write your description
    """
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    return src

class Render(torch.nn.Module):
    def __init__(self, renderer):
        """
        Initialize the renderer.

        Args:
            self: (todo): write your description
            renderer: (bool): write your description
        """
        super(Render, self).__init__()
        self.renderer = renderer

    def forward(self, vertices, faces, textures=None):
        """
        Forward vertices

        Args:
            self: (todo): write your description
            vertices: (array): write your description
            faces: (todo): write your description
            textures: (todo): write your description
        """
        vs = vertices
        vs[:, :, 1] *= -1
        fs = faces
        if(textures is None):
            mesh_ = sr.Mesh(vs, fs)
        else:
            ts = textures
            mesh_ = sr.Mesh(vs, fs, ts)
        imgs = self.renderer.render_mesh(mesh_)
        return imgs

########################################################################
############## Wrapper torch module for Neural Renderer ################
########################################################################
class SoftRenderer(torch.nn.Module):
    """
    This is the core pytorch function to call.
    """
    def __init__(self, img_size=256, render_type='softmax', background_color=[0,0,0], sigma_val=1e-5, gamma_val=1e-4, dist_eps=1e-10, anti_aliasing=True):
        """
        Initialize an image.

        Args:
            self: (todo): write your description
            img_size: (int): write your description
            render_type: (str): write your description
            background_color: (bool): write your description
            sigma_val: (str): write your description
            gamma_val: (float): write your description
            dist_eps: (float): write your description
            anti_aliasing: (str): write your description
        """
        super(SoftRenderer, self).__init__()

        self.renderer = sr.SoftRenderer(image_size=img_size, aggr_func_rgb=render_type, camera_mode='look_at', sigma_val=sigma_val, dist_eps=dist_eps, gamma_val=gamma_val, background_color=background_color, anti_aliasing=anti_aliasing, perspective=False)


        # Set a default camera to be at (0, 0, -2.732)
        self.renderer.transform.transformer._eye = [0, 0, -2.732]

        # Make it a bit brighter for vis
        self.renderer.lighting.ambient.light_intensity = 0.8

        self.proj_fn = geom_utils.orthographic_proj_withz
        self.offset_z = 5.

    def ambient_light_only(self):
        """
        Ambient light light.

        Args:
            self: (todo): write your description
        """
        # Make light only ambient.
        self.renderer.lighting.ambient.light_intensity = 1
        self.renderer.lighting.directionals[0].light_intensity = 0

    def set_bgcolor(self, color):
        """
        Set the background color.

        Args:
            self: (todo): write your description
            color: (str): write your description
        """
        self.renderer.rasterizer.background_color = color

    def project_points(self, verts, cams):
        """
        Return points on the projection.

        Args:
            self: (todo): write your description
            verts: (str): write your description
            cams: (todo): write your description
        """
        proj = self.proj_fn(verts, cams)
        return proj[:, :, :2]

    def forward(self, vertices, faces, cams, textures=None):
        """
        Forward vertices in vertices.

        Args:
            self: (todo): write your description
            vertices: (array): write your description
            faces: (todo): write your description
            cams: (todo): write your description
            textures: (todo): write your description
        """
        faces = faces.int()
        verts = self.proj_fn(vertices, cams, offset_z=self.offset_z)

        if textures is not None:
            return Render(self.renderer)(verts, faces, textures)
        else:
            return Render(self.renderer)(verts, faces)

########################################################################
############################## Tests ###################################
########################################################################
def teapot_deform_test():
    """
    Teapototot image.

    Args:
    """
    obj_file = '../external/neural_renderer/examples/data/teapot.obj'
    img_file = '../external/neural_renderer/examples/data/example2_ref.png'
    img_save_dir = '../cachedir/nmr/'

    mesh = sr.Mesh.from_obj(obj_file)
    vertices = mesh.vertices
    faces = mesh.faces

    image_ref = scipy.misc.imread(img_file).astype('float32').mean(-1) / 255.
    image_ref = torch.Tensor(image_ref[None, :, :]).cuda(device=0)

    mask_renderer = SoftRenderer()
    faces_var = faces.cuda(device=0)
    cams = np.array([1., 0, 0, 1, 0, 0, 0], dtype=np.float32)
    cams_var = torch.from_numpy(cams[None, :]).cuda(device=0)

    class TeapotModel(torch.nn.Module):
        def __init__(self):
            """
            Initialize the device.

            Args:
                self: (todo): write your description
            """
            super(TeapotModel, self).__init__()
            vertices_var = vertices.cuda(device=0)
            self.vertices_var = torch.nn.Parameter(vertices_var)

        def forward(self):
            """
            Forward computation.

            Args:
                self: (todo): write your description
            """
            tmp = mask_renderer.forward(self.vertices_var, faces_var, cams_var)
            return tmp

    opt_model = TeapotModel()
    optimizer = torch.optim.Adam(opt_model.parameters(), lr=1e-2, betas=(0.9, 0.999))
    # from time import time
    loop = tqdm.tqdm(range(300))
    print('Optimizing Vertices: ')
    for ix in loop:
        optimizer.zero_grad()
        masks_pred = opt_model.forward()
        loss = torch.nn.MSELoss()(masks_pred, image_ref)
        loss.backward()
        if ix % 20 == 0:
            im_rendered = masks_pred.data.cpu().numpy()[0, :, :]
            scipy.misc.imsave(img_save_dir + 'iter_{}.png'.format(ix), im_rendered)
        optimizer.step()

if __name__ == '__main__':
    teapot_deform_test()
