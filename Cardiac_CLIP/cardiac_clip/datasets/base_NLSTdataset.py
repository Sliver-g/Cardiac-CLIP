import io
import os
import random

import pyarrow as pa
import torch
from PIL import Image
####################
import numpy as np
import json
import SimpleITK as sitk
from scipy import ndimage
####################
#from ..transforms import keys_to_transforms

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


class BaseDatasetNLST(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str,
            transform_keys: list,
            image_size: int,
            names: list,
            text_column_name: str = "",
            max_text_len: int = 40,
            draw_false_image: int = 0,
            draw_false_text: int = 0,
            image_only: bool = False,
            label_column_name: str = "",
    ):
        super().__init__()
        assert len(transform_keys) >= 1
        # Hyper-Parameters
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.label_column_name = label_column_name
        
        with open(data_dir)as f:
            nlst_data = json.load(f)
            tables=[]
            
            for name in names:
                if name == 'train':
                    #1%, 10%, 100% finetune
                    train_data = nlst_data[name]
                    sample_size = max(1, int(len(train_data) * 1))
                    sampled_train_data = train_data[:sample_size]
                    tables.append(sampled_train_data)
                else:
                    # for validation and test
                    tables.append(nlst_data[name])

            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])
            self.table = [element for sublist in tables for element in sublist]


    def __len__(self):
        return len(self.table)
    
    def get_label(self,index):
        label=self.table[index]['label']
        binary_list=[label]
        return {
            'labels':binary_list
        }
    

    def get_raw_image(self, index):
        img_path=self.table[index]['img_path']
        #image_npy = np.load(img_path).copy()
        sitk_image = sitk.ReadImage(img_path) #用于ACS
        image_npy = sitk.GetArrayFromImage(sitk_image)
        image_npy = normalization(image_npy)

        return image_npy

    def get_image(self, index):
        image_array = self.get_raw_image(index)
        image_tensor=torch.from_numpy(image_array)
        #image_tensor=image_tensor.unsqueeze(0)
        return {
            "image": image_tensor,
            "raw_index": index,
        }

    def get_suite(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                ret.update(self.get_label(index))
                txt = 'nothing'
                #ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update({'text':txt})
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
        return ret

    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            #img_sizes += [ii.shape for i in img if i is not None for ii in i]
            img_sizes += [i.shape for i in img if i is not None]

        if len(img_keys) != 0:
            max_channel = max([i[0]for i in img_sizes])
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]
            view_size = len(img[0])
            new_images = torch.zeros(batch_size,1, max_channel, max_height, max_width)
            for bi in range(batch_size):
                orig = img[bi]
                new_images[bi, :, : orig.shape[0], : orig.shape[1],:orig.shape[2]] = orig
            dict_batch[img_key] = new_images
            

        return dict_batch
