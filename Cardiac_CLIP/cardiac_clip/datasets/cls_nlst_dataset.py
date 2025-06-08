from .base_NLSTdataset import BaseDatasetNLST
#import torch


class NLSTDataset(BaseDatasetNLST):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names=['train']
        elif split == "val":
            names=['test']
        elif split == "test":
            names=['test']
        else:
            raise ValueError

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
   
    def __getitem__(self, index):
        return self.get_suite(index)