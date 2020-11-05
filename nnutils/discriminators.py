# -----------------------------------------------------------------------------------
# Code adapted from: 
# https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
# 
# MIT License
# 
# Copyright (c) 2018 Joris
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        """
        Forward forward forward.

        Args:
            ctx: (todo): write your description
            x: (todo): write your description
            lambda_: (todo): write your description
        """
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        """
        Computes the backward backward.

        Args:
            ctx: (todo): write your description
            grads: (todo): write your description
        """
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        """
        Initialize the superclass.

        Args:
            self: (todo): write your description
            lambda_: (float): write your description
        """
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        """
        Evaluate of x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return GradientReversalFunction.apply(x, self.lambda_)


class Discriminator(nn.Module):
    def __init__(self, lambda_, in_dim=1, img_size=64):
        """
        Initialize the ndarray.

        Args:
            self: (todo): write your description
            lambda_: (float): write your description
            in_dim: (int): write your description
            img_size: (int): write your description
        """
        super(Discriminator, self).__init__()
        fc_size = int(img_size // 16)
        self.grl = GradientReversal(lambda_ = lambda_)
        # 64 x 64
        self.img_conv = nn.Conv2d(in_dim, 32, 3, 2, 1)
        # 32 x 32
        self.convs = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1),
                                   nn.ReLU(True),
                                   nn.Conv2d(64, 32, 3, 2, 1),
                                   nn.ReLU(True),
                                   nn.Conv2d(32, 32, 3, 2, 1),
                                   nn.ReLU(True),
                                   nn.Conv2d(32, 1, 1, 1, 0))
        self.fc = nn.Linear(fc_size*fc_size, 1)
        self.in_dim = in_dim

    def forward(self, imgs):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            imgs: (todo): write your description
        """
        b = imgs.size(0)
        imgs_grl = self.grl(imgs)
        conv_feat = F.relu(self.img_conv(imgs_grl))
        # b * 32 * 32 * 32
        pmap = self.convs(conv_feat)
        pmap = pmap.view(b,-1)
        return self.fc(pmap)
