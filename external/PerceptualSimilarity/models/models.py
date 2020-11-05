
def create_model(opt):
    """
    Vel factory function

    Args:
        opt: (str): write your description
    """
    model = None
    print(opt.model)
    from .siam_model import *
    model = DistModel()
    model.initialize(opt, opt.batchSize, )
    print("model [%s] was created" % (model.name()))
    return model

