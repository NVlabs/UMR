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
def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.log_name = os.path.join(opt.checkpoint_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # scalars: same format as |scalars| of plot_current_scalars
    def print_current_scalars(self, epoch, i, scalars):
        message = green('(epoch: %d, iters: %d) ' % (epoch, i))
        for k, v in scalars.items():
            if("lr" in k):
                message += '%s: %.6f ' % (k, v)
            else:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
