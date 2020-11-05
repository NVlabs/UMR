import torch.utils.data
from data.base_data_loader import BaseDataLoader
import os

def CreateDataset(dataroots,dataset_mode='2afc',load_size=64,):
    """
    Initialize a dataset.

    Args:
        dataroots: (str): write your description
        dataset_mode: (str): write your description
        load_size: (int): write your description
    """
    dataset = None
    if dataset_mode=='2afc': # human judgements
        from dataset.twoafc_dataset import TwoAFCDataset
        dataset = TwoAFCDataset()
    elif dataset_mode=='jnd': # human judgements
        from dataset.jnd_dataset import JNDDataset
        dataset = JNDDataset()
    else:
        raise ValueError("Dataset Mode [%s] not recognized."%self.dataset_mode)

    dataset.initialize(dataroots,load_size=load_size)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        """
        Return the name for this node.

        Args:
            self: (todo): write your description
        """
        return 'CustomDatasetDataLoader'

    def initialize(self, datafolders, dataroot='./dataset',dataset_mode='2afc',load_size=64,batch_size=1,serial_batches=True, nThreads=1):
        """
        Initializes the dataset.

        Args:
            self: (todo): write your description
            datafolders: (todo): write your description
            dataroot: (array): write your description
            dataset_mode: (todo): write your description
            load_size: (int): write your description
            batch_size: (int): write your description
            serial_batches: (todo): write your description
            nThreads: (int): write your description
        """
        BaseDataLoader.initialize(self)
        if(not isinstance(datafolders,list)):
            datafolders = [datafolders,]
        data_root_folders = [os.path.join(dataroot,datafolder) for datafolder in datafolders]
        self.dataset = CreateDataset(data_root_folders,dataset_mode=dataset_mode,load_size=load_size)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=not serial_batches,
            num_workers=int(nThreads))

    def load_data(self):
        """
        Load the data from the data.

        Args:
            self: (todo): write your description
        """
        return self.dataloader

    def __len__(self):
        """
        Returns the number of the dataset.

        Args:
            self: (todo): write your description
        """
        return len(self.dataset)
