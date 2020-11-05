
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

import soft_renderer as sr


class Renderer(nn.Module):
    def __init__(self, image_size=256, background_color=[0,0,0], near=1, far=100, 
                 anti_aliasing=True, fill_back=True, eps=1e-6,
                 camera_mode='projection',
                 P=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1],
                 light_mode='surface',
                 light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                 light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                 light_directions=[0,1,0]):
        """
        Initialize the image.

        Args:
            self: (todo): write your description
            image_size: (int): write your description
            background_color: (bool): write your description
            near: (int): write your description
            far: (int): write your description
            anti_aliasing: (str): write your description
            fill_back: (str): write your description
            eps: (float): write your description
            camera_mode: (str): write your description
            P: (int): write your description
            dist_coeffs: (str): write your description
            orig_size: (int): write your description
            perspective: (todo): write your description
            viewing_angle: (todo): write your description
            viewing_scale: (todo): write your description
            eye: (str): write your description
            camera_direction: (str): write your description
            light_mode: (str): write your description
            light_intensity_ambient: (str): write your description
            light_color_ambient: (todo): write your description
            light_intensity_directionals: (str): write your description
            light_color_directionals: (str): write your description
            light_directions: (str): write your description
        """
        super(Renderer, self).__init__()

        # light
        self.lighting = sr.Lighting(light_mode,
                                    light_intensity_ambient, light_color_ambient,
                                    light_intensity_directionals, light_color_directionals,
                                    light_directions)

        # camera
        self.transform = sr.Transform(camera_mode, 
                                      P, dist_coeffs, orig_size,
                                      perspective, viewing_angle, viewing_scale, 
                                      eye, camera_direction)

        # rasterization
        self.rasterizer = sr.Rasterizer(image_size, background_color, near, far, 
                                        anti_aliasing, fill_back, eps)

    def forward(self, mesh, mode=None):
        """
        Raster

        Args:
            self: (todo): write your description
            mesh: (todo): write your description
            mode: (str): write your description
        """
        mesh = self.lighting(mesh)
        mesh = self.transform(mesh)
        return self.rasterizer(mesh, mode)


class SoftRenderer(nn.Module):
    def __init__(self, image_size=256, background_color=[0,0,0], near=1, far=100, 
                 anti_aliasing=False, fill_back=True, eps=1e-3,
                 sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-4,
                 gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                 texture_type='surface',
                 camera_mode='projection',
                 P=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1],
                 light_mode='surface',
                 light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                 light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                 light_directions=[0,1,0]):
        """
        Initialize the image.

        Args:
            self: (todo): write your description
            image_size: (int): write your description
            background_color: (bool): write your description
            near: (int): write your description
            far: (int): write your description
            anti_aliasing: (str): write your description
            fill_back: (str): write your description
            eps: (float): write your description
            sigma_val: (str): write your description
            dist_func: (todo): write your description
            dist_eps: (float): write your description
            gamma_val: (float): write your description
            aggr_func_rgb: (todo): write your description
            aggr_func_alpha: (todo): write your description
            texture_type: (str): write your description
            camera_mode: (str): write your description
            P: (int): write your description
            dist_coeffs: (str): write your description
            orig_size: (int): write your description
            perspective: (todo): write your description
            viewing_angle: (todo): write your description
            viewing_scale: (todo): write your description
            eye: (str): write your description
            camera_direction: (str): write your description
            light_mode: (str): write your description
            light_intensity_ambient: (str): write your description
            light_color_ambient: (todo): write your description
            light_intensity_directionals: (str): write your description
            light_color_directionals: (str): write your description
            light_directions: (str): write your description
        """
        super(SoftRenderer, self).__init__()

        # light
        self.lighting = sr.Lighting(light_mode,
                                    light_intensity_ambient, light_color_ambient,
                                    light_intensity_directionals, light_color_directionals,
                                    light_directions)

        # camera
        self.transform = sr.Transform(camera_mode, 
                                      P, dist_coeffs, orig_size,
                                      perspective, viewing_angle, viewing_scale, 
                                      eye, camera_direction)

        # rasterization
        self.rasterizer = sr.SoftRasterizer(image_size, background_color, near, far, 
                                            anti_aliasing, fill_back, eps,
                                            sigma_val, dist_func, dist_eps,
                                            gamma_val, aggr_func_rgb, aggr_func_alpha,
                                            texture_type)

    def set_sigma(self, sigma):
        """
        Set sigma

        Args:
            self: (todo): write your description
            sigma: (float): write your description
        """
        self.rasterizer.sigma_val = sigma

    def set_gamma(self, gamma):
        """
        Set gamma gamma function.

        Args:
            self: (todo): write your description
            gamma: (todo): write your description
        """
        self.rasterizer.gamma_val = gamma

    def set_texture_mode(self, mode):
        """
        Set the texture mode.

        Args:
            self: (todo): write your description
            mode: (str): write your description
        """
        assert mode in ['vertex', 'surface'], 'Mode only support surface and vertex'

        self.lighting.light_mode = mode
        self.rasterizer.texture_type = mode

    def render_mesh(self, mesh, mode=None):
        """
        Render a mesh as a : class : raster.

        Args:
            self: (todo): write your description
            mesh: (todo): write your description
            mode: (str): write your description
        """
        self.set_texture_mode(mesh.texture_type)
        mesh = self.lighting(mesh)
        mesh = self.transform(mesh)
        return self.rasterizer(mesh, mode)

    def forward(self, vertices, faces, textures=None, mode=None, texture_type='surface'):
        """
        Render a list of the vertices.

        Args:
            self: (todo): write your description
            vertices: (array): write your description
            faces: (todo): write your description
            textures: (todo): write your description
            mode: (str): write your description
            texture_type: (str): write your description
        """
        mesh = sr.Mesh(vertices, faces, textures=textures, texture_type=texture_type)
        return self.render_mesh(mesh, mode)