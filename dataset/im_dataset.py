from utils import DiskIODataset


class ImTrainingSet(DiskIODataset):
    def __init__(self, opts):
        opts['if_train'] = True
        super().__init__(**opts)


class ImTestSet(DiskIODataset):
    def __init__(self, opts):
        opts['if_train'] = False
        super().__init__(**opts)
