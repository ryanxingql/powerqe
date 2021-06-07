import net

from utils import BaseAlg


class ARCNNAlgorithm(BaseAlg):
    def __init__(self, opts_dict, if_train, if_dist):
        model_cls = getattr(net, 'ARCNNModel')  # !!!
        super().__init__(opts_dict=opts_dict, model_cls=model_cls, if_train=if_train, if_dist=if_dist)
