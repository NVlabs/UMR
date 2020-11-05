def CreateDataLoader(datafolder,dataroot='./dataset',dataset_mode='2afc',load_size=64,batch_size=1,serial_batches=True):
    """
    Creates dataset.

    Args:
        datafolder: (todo): write your description
        dataroot: (todo): write your description
        dataset_mode: (todo): write your description
        load_size: (int): write your description
        batch_size: (int): write your description
        serial_batches: (todo): write your description
    """
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    # print(data_loader.name())
    data_loader.initialize(datafolder,dataroot=dataroot+'/'+dataset_mode,dataset_mode=dataset_mode,load_size=load_size,batch_size=batch_size,serial_batches=serial_batches, nThreads=1)
    return data_loader
