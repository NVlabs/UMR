import os
import torch
from ..util import util as util
from torch.autograd import Variable
from pdb import set_trace as st
from IPython import embed

class BaseModel():
    def __init__(self):
        """
        Initialize the object

        Args:
            self: (todo): write your description
        """
        pass;
        
    def name(self):
        """
        Return the name for this node.

        Args:
            self: (todo): write your description
        """
        return 'BaseModel'

    def initialize(self, use_gpu=True):
        """
        Initialize the tensorboard.

        Args:
            self: (todo): write your description
            use_gpu: (bool): write your description
        """
        self.use_gpu = use_gpu
        self.Tensor = torch.cuda.FloatTensor if self.use_gpu else torch.Tensor
        # self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def forward(self):
        """
        Forward the next call todo is_self.

        Args:
            self: (todo): write your description
        """
        pass

    def get_image_paths(self):
        """
        Get the list of image paths.

        Args:
            self: (todo): write your description
        """
        pass

    def optimize_parameters(self):
        """
        Optimize the parameters.

        Args:
            self: (todo): write your description
        """
        pass

    def get_current_visuals(self):
        """
        Get visual visual visual visual visual visual visual

        Args:
            self: (todo): write your description
        """
        return self.input

    def get_current_errors(self):
        """
        Returns a dict of errors.

        Args:
            self: (todo): write your description
        """
        return {}

    def save(self, label):
        """
        Save the given label

        Args:
            self: (todo): write your description
            label: (str): write your description
        """
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, path, network_label, epoch_label):
        """
        Save network to disk to disk.

        Args:
            self: (todo): write your description
            network: (todo): write your description
            path: (str): write your description
            network_label: (todo): write your description
            epoch_label: (todo): write your description
        """
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        """
        Load network from disk.

        Args:
            self: (todo): write your description
            network: (todo): write your description
            network_label: (str): write your description
            epoch_label: (todo): write your description
        """
        # embed()
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print('Loading network from %s'%save_path)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate():
        """
        Update the learning rate.

        Args:
        """
        pass

    def get_image_paths(self):
        """
        : return : a list of the image paths

        Args:
            self: (todo): write your description
        """
        return self.image_paths

    def save_done(self, flag=False):
        """
        Save the flag todo file.

        Args:
            self: (todo): write your description
            flag: (todo): write your description
        """
        np.save(os.path.join(self.save_dir, 'done_flag'),flag)
        np.savetxt(os.path.join(self.save_dir, 'done_flag'),[flag,],fmt='%i')

