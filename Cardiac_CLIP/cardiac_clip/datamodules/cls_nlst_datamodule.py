from .base_datamodule import BaseDataModule
from ..datasets import NLSTDataset
import random
from torch.utils.data import Sampler
from torch.utils.data import DataLoader


class NLSTDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return NLSTDataset

    @property
    def dataset_cls_no_false(self):
        return NLSTDataset

    @property
    def dataset_name(self):
        return "NLST"

