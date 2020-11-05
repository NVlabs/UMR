
import math
import numpy as np
import torch
import torch.nn as nn

import soft_renderer.functional as srf

class Projection(nn.Module):
    def __init__(self, P, dist_coeffs=None, orig_size=512):
        """
        Initialize transition function.

        Args:
            self: (todo): write your description
            P: (int): write your description
            dist_coeffs: (str): write your description
            orig_size: (int): write your description
        """
        super(Projection, self).__init__()

        self.P = P
        self.dist_coeffs = dist_coeffs
        self.orig_size = orig_size

        if isinstance(self.P, np.ndarray):
            self.P = torch.from_numpy(self.P).cuda()
        if self.P is None or P.ndimension() != 3 or self.P.shape[1] != 3 or self.P.shape[2] != 4:
            raise ValueError('You need to provide a valid (batch_size)x3x4 projection matrix')
        if dist_coeffs is None:
            self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]).repeat(P.shape[0], 1)

    def forward(self, vertices):
        """
        Forward vertices.

        Args:
            self: (todo): write your description
            vertices: (array): write your description
        """
        vertices = srf.projection(vertices, self.P, self.dist_coeffs, self.orig_size)
        return vertices


class LookAt(nn.Module):
    def __init__(self, perspective=True, viewing_angle=30, viewing_scale=1.0, eye=None):
        """
        Initialize the angle.

        Args:
            self: (todo): write your description
            perspective: (todo): write your description
            viewing_angle: (todo): write your description
            viewing_scale: (todo): write your description
            eye: (str): write your description
        """
        super(LookAt, self).__init__()

        self.perspective = perspective
        self.viewing_angle = viewing_angle
        self.viewing_scale = viewing_scale
        self._eye = eye

        if self._eye is None:
            self._eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]

    def forward(self, vertices):
        """
        Forward vertices.

        Args:
            self: (todo): write your description
            vertices: (array): write your description
        """
        vertices = srf.look_at(vertices, self._eye)
        # perspective transformation
        if self.perspective:
            vertices = srf.perspective(vertices, angle=self.viewing_angle)
        else:
            vertices = srf.orthogonal(vertices, scale=self.viewing_scale)
        return vertices


class Look(nn.Module):
    def __init__(self, camera_direction=[0,0,1], perspective=True, viewing_angle=30, viewing_scale=1.0, eye=None):
        """
        Initialize the camera.

        Args:
            self: (todo): write your description
            camera_direction: (str): write your description
            perspective: (todo): write your description
            viewing_angle: (todo): write your description
            viewing_scale: (todo): write your description
            eye: (str): write your description
        """
        super(Look, self).__init__()
        
        self.perspective = perspective
        self.viewing_angle = viewing_angle
        self.viewing_scale = viewing_scale
        self._eye = eye
        self.camera_direction = [0, 0, 1]

        if self._eye is None:
            self._eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]

    def forward(self, vertices):
        """
        Forward the camera vertices.

        Args:
            self: (todo): write your description
            vertices: (array): write your description
        """
        vertices = srf.look(vertices, self._eye, self.camera_direction)
        # perspective transformation
        if self.perspective:
            vertices = srf.perspective(vertices, angle=self.viewing_angle)
        else:
            vertices = srf.orthogonal(vertices, scale=self.viewing_scale)
        return vertices


class Transform(nn.Module):
    def __init__(self, camera_mode='projection', P=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0, 
                 eye=None, camera_direction=[0,0,1]):
        """
        Initialize the camera.

        Args:
            self: (todo): write your description
            camera_mode: (str): write your description
            P: (int): write your description
            dist_coeffs: (str): write your description
            orig_size: (int): write your description
            perspective: (todo): write your description
            viewing_angle: (todo): write your description
            viewing_scale: (todo): write your description
            eye: (str): write your description
            camera_direction: (str): write your description
        """
        super(Transform, self).__init__()

        self.camera_mode = camera_mode
        if self.camera_mode == 'projection':
            self.transformer = Projection(P, dist_coeffs, orig_size)
        elif self.camera_mode == 'look':
            self.transformer = Look(perspective, viewing_angle, viewing_scale, eye, camera_direction)
        elif self.camera_mode == 'look_at':
            self.transformer = LookAt(perspective, viewing_angle, viewing_scale, eye)
        else:
            raise ValueError('Camera mode has to be one of projection, look or look_at')

    def forward(self, mesh):
        """
        Forward mesh. mesh.

        Args:
            self: (todo): write your description
            mesh: (todo): write your description
        """
        mesh.vertices = self.transformer(mesh.vertices)
        return mesh

    def set_eyes_from_angles(self, distances, elevations, azimuths):
        """
        Sets the camera angle of the camera.

        Args:
            self: (todo): write your description
            distances: (todo): write your description
            elevations: (todo): write your description
            azimuths: (str): write your description
        """
        if self.camera_mode not in ['look', 'look_at']:
            raise ValueError('Projection does not need to set eyes')
        self.transformer._eye = srf.get_points_from_angles(distances, elevations, azimuths)

    def set_eyes(self, eyes):
        """
        Set the camera mode.

        Args:
            self: (todo): write your description
            eyes: (str): write your description
        """
        if self.camera_mode not in ['look', 'look_at']:
            raise ValueError('Projection does not need to set eyes')
        self.transformer._eye = eyes

    @property
    def eyes(self):
        """
        : class : py : class.

        Args:
            self: (todo): write your description
        """
        return self.transformer._eyes
    