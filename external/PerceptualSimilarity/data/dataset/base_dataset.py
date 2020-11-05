import torch.utils.data as data

class BaseDataset(data.Dataset):
    def __init__(self):
        """
        Initialize the dataset.

        Args:
            self: (todo): write your description
        """
        super(BaseDataset, self).__init__()
        
    def name(self):
        """
        Return the name for this node.

        Args:
            self: (todo): write your description
        """
        return 'BaseDataset'
    
    def initialize(self):
        """
        Initialize the next callable object

        Args:
            self: (todo): write your description
        """
        pass

