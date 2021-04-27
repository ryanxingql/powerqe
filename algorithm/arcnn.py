import net
from utils import BaseAlg

class ARCNNAlgorithm(BaseAlg):
    """use most of the BaseAlg functions."""
    def __init__(self, opts_dict, if_train):
        self.opts_dict = opts_dict
        self.if_train = if_train

        model_cls = getattr(net, 'ARCNNModel')
        self.create_model(
            model_cls=model_cls,
            opts_dict=self.opts_dict['network'],
            if_train=self.if_train,
        )

        super().__init__()  # to further obtain optim, loss, etc.
