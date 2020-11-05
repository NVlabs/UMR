# -----------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Xueting Li.
# -----------------------------------------------------------
import numpy as np
import os
import ntpath
import time
import termcolor

# convert to colored strings
"""
Return color

Args:
    content: (todo): write your description
"""
def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
"""
Returns a string representing a string

Args:
    content: (todo): write your description
"""
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
"""
Generate a string.

Args:
    content: (str): write your description
"""
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
"""
Generate a string

Args:
    content: (str): write your description
"""
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
"""
Returns html string.

Args:
    content: (str): write your description
"""
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
"""
Draw a color

Args:
    content: (str): write your description
"""
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])

class Visualizer():
    def __init__(self, opt):
        """
        Initialize the log file.

        Args:
            self: (todo): write your description
            opt: (dict): write your description
        """
        # self.opt = opt
        self.log_name = os.path.join(opt.checkpoint_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # scalars: same format as |scalars| of plot_current_scalars
    def print_current_scalars(self, epoch, i, scalars):
        """
        Print the current velocity.

        Args:
            self: (todo): write your description
            epoch: (int): write your description
            i: (todo): write your description
            scalars: (dict): write your description
        """
        message = green('(epoch: %d, iters: %d) ' % (epoch, i))
        for k, v in scalars.items():
            if("lr" in k):
                message += '%s: %.6f ' % (k, v)
            else:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
