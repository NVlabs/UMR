################################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
################################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

NP_EXTENSIONS = ['.npy',]

def is_image_file(filename, mode='img'):
    """
    Check if a file is an image file.

    Args:
        filename: (str): write your description
        mode: (str): write your description
    """
    if(mode=='img'):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    elif(mode=='np'):
        return any(filename.endswith(extension) for extension in NP_EXTENSIONS)

def make_dataset(dirs, mode='img'):
    """
    Make a list of images.

    Args:
        dirs: (str): write your description
        mode: (str): write your description
    """
    if(not isinstance(dirs,list)):
        dirs = [dirs,]

    images = []
    for dir in dirs:
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname, mode=mode):
                    path = os.path.join(root, fname)
                    images.append(path)

    # print("Found %i images in %s"%(len(images),root))
    return images

def default_loader(path):
    """
    Loads an image.

    Args:
        path: (str): write your description
    """
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        """
        Initialize the image.

        Args:
            self: (todo): write your description
            root: (str): write your description
            transform: (str): write your description
            return_paths: (str): write your description
            loader: (todo): write your description
            default_loader: (str): write your description
        """
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        """
        Get an item from an index

        Args:
            self: (todo): write your description
            index: (int): write your description
        """
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        """
        Returns the number of bytes in - place.

        Args:
            self: (todo): write your description
        """
        return len(self.imgs)
