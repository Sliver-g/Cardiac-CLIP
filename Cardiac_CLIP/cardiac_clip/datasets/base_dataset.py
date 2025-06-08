import io
import os
import random

import pyarrow as pa
import torch
from PIL import Image
####################
import numpy as np
import json
####################
#from ..transforms import keys_to_transforms

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


class BaseDataset(torch.utils.data.Dataset):
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

        # Read Texts
        if len(names) != 0:
            with open(data_dir)as f:
                jun_data = json.load(f)
                tables=[jun_data[name]
                        for name in names
                        ]

                self.table_names = list()
                for i, name in enumerate(names):
                    self.table_names += [name] * len(tables[i])

                self.table = [element for sublist in tables for element in sublist]
                if text_column_name != "":
                    self.text_column_name = text_column_name
                    self.all_texts = [text for item in self.table for text in item['texts']]
                else:
                    self.all_texts = list()
        else:
            self.all_texts = list()


        self.index_mapper = dict()
        if text_column_name != "" and not self.image_only:
            j = 0
            for index, item in enumerate(self.table):
                for i,img in enumerate(item['img_path']):
                    for _j,txt in enumerate(item['texts']):
                        self.index_mapper[j] = (index,i, _j) 
                        j += 1


    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper)
        #return len(self.table)
        
    def get_label(self,index):# get soft label
        index, image_index, caption_index = self.index_mapper[index]
        labels=self.table[index]['labels']
        return {
            "labels":labels
        }
    
    def get_raw_image(self, index, image_key="image"):# 改
        index, image_index, caption_index = self.index_mapper[index]
        image_npy=np.load(self.table[index]['img_path'][image_index])
        image_npy = normalization(image_npy)
        return image_npy

    def get_image(self, index, image_key="image"):
        image_array = self.get_raw_image(index, image_key=image_key)
        #image_tensor = [tr(image) for tr in self.transforms]
        image_tensor=torch.from_numpy(image_array)
        return {
            "image": image_tensor,
            #"img_index": self.index_mapper[index][0],
            #"cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }

    def get_false_image(self, rep, image_key="image", selected_index=None):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        #random_index = random.randint(0, len(self.table) - 1)
        image_array = self.get_raw_image(random_index, image_key=image_key)
        #image_tensor = [tr(image) for tr in self.transforms]
        image_tensor=torch.from_numpy(image_array)
        return {f"false_image_{rep}": image_tensor}

    def get_text(self, raw_index):
        index, image_index, caption_index = self.index_mapper[raw_index]
        text =self.table[index]['texts'][caption_index]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        return {
            "text": (text, encoding),
            "raw_index": raw_index,
        }

    def get_false_text(self, rep, selected_index=None):
        #random_index = random.randint(0, len(self.index_mapper) - 1)
        random_index = random.randint(0, len(self.table) - 1)
        #index, caption_index = self.index_mapper[random_index]
        #text = self.all_texts[index][caption_index]
        text_list=self.table[random_index]['texts']
        text=random.choice(text_list)
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                ret.update(self.get_label(index))#获取软标签
                if not self.image_only:
                    txt = self.get_text(index)
                    #ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    ret.update(txt)
                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i, selected_index=index))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i, selected_index=index))
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                #index = random.randint(0, len(self.index_mapper) - 1)
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
            img_sizes += [i.shape for i in img if i is not None] #改

   
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
                new_images[bi, :, : orig.shape[0], : orig.shape[1],:orig.shape[2]] = orig ####
            dict_batch[img_key] = new_images
            
        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
        if len(txt_keys) != 0:
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = ([d[0] for d in dict_batch[txt_key]], [d[1] for d in dict_batch[txt_key]])
                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i): batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i): batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask


        return dict_batch
