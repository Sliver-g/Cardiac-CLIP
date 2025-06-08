import functools

import torch
import torch.nn.functional as F
import tqdm
from einops import rearrange
from torch.utils.data.distributed import DistributedSampler

from .dist_utils import all_gather
import numpy as np
import torch.nn as nn


def compute_mim(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)

    if pl_module.hparams.config["mim_layer"] == -1:
        multi_modal_image_feats = infer["multi_modal_image_feats"]
    else:
        layer_idx = pl_module.hparams.config["mim_layer"]
        multi_modal_image_feats = infer[f"multi_modal_image_feats_{layer_idx}"]

    mim_logits = pl_module.mim_head(multi_modal_image_feats, infer["mim_ids_restore"])

    target = infer["patched_images"]
    if pl_module.hparams.config["norm_pix_loss"]:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5
    mim_labels = target
    mask = infer["mim_masks"]

    mim_loss = (mim_logits - mim_labels) ** 2
    mim_loss = mim_loss.mean(dim=-1)  # [N, L], mean loss per patch
    mim_loss = (mim_loss * mask).sum() / mask.sum()  # mean loss on removed patches

    ret = {
        "mim_loss": mim_loss,
        "mim_logits": mim_logits,
        "mim_labels": mim_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mim_loss")(ret["mim_loss"])
    acc = -loss
    pl_module.log(f"mim/{phase}/loss", loss)
    pl_module.log(f"mim/{phase}/accuracy", acc)

    return ret

#Loss of clip
def compute_itc(pl_module,batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    image_features=infer["multi_modal_image_feats"]
    text_features = infer["multi_modal_text_feats"]

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logit_scale=pl_module.vision_encoder.logit_scale
    #logit_scale=pl_module.model.model.logit_scale
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t() 

    # labels = torch.arange(len(logits_per_image)).to(pl_module.device)

    # image_loss = F.cross_entropy(logits_per_image, labels)
    # text_loss  = F.cross_entropy(logits_per_text, labels)
    
    '''--------------------soft label----------------------------'''
    labels_array=np.array(batch['labels'])
    batch_size, num_classes = labels_array.shape

    # 
    soft_labels = np.zeros((batch_size, batch_size))

    # cosine similarity
    for i in range(batch_size):
        for j in range(batch_size):
            # calculate
            dot_product = np.dot(labels_array[i], labels_array[j]) 
            norm_i = np.linalg.norm(labels_array[i])
            norm_j = np.linalg.norm(labels_array[j]) 
            
            similarity = dot_product / (norm_i * norm_j) # 计算余弦相似度
                
            soft_labels[i, j] = similarity
    
    soft_labels=torch.from_numpy(soft_labels)
    soft_labels=soft_labels.to(logits_per_image.device)
    # softmax
    soft_labels = F.softmax(soft_labels, dim=1)
    soft_labels_text=soft_labels.t()
            
    image_probabilities = torch.nn.functional.log_softmax(logits_per_image, dim=1)
    image_loss = -torch.sum(soft_labels * image_probabilities, dim=1).mean() 
    
    text_probabilities = torch.nn.functional.log_softmax(logits_per_text, dim=1)
    text_loss = -torch.sum(soft_labels_text * text_probabilities, dim=1).mean()   
    '''-----------------------------------------------------'''
    
    loss = (image_loss + text_loss) / 2

    ret = {
        "itc_loss": loss,
        #"itc_labels": labels,
        "itc_soft_labels":soft_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itc_loss")(ret["itc_loss"])
    #acc = getattr(pl_module, f"{phase}_itc_accuracy")(ret["mlm_logits"], ret["mlm_labels"])
    acc = -loss
    pl_module.log(f"itc/{phase}/loss", loss)
    pl_module.log(f"itc/{phase}/accuracy", acc)

    return ret


#classification loss
def compute_cls(pl_module, batch, test=False):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    cls_logits = infer["multi_modal_cls_feats"] 

    auroc_logits=torch.sigmoid(cls_logits)

    cls_labels = batch["labels"]
    #转化为张量
    all_labels=torch.tensor(cls_labels,dtype=torch.float).to(auroc_logits.device)
    all_labels[all_labels == -1] = 0

    cls_loss = F.binary_cross_entropy_with_logits(cls_logits, all_labels)

    ret = {
        "cls_loss": cls_loss,
        "cls_logits": cls_logits,
        "cls_labels": cls_labels,
        "auroc_logits":auroc_logits,
    }

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    loss = getattr(pl_module, f"{phase}_cls_loss")(ret["cls_loss"])
    
    auroc = getattr(pl_module, f"{phase}_cls_auroc")(auroc_logits, all_labels)

    pl_module.log(f"cls/{phase}/loss", loss)
    pl_module.log(f"cls/{phase}/auroc", auroc)


    return ret


def compute_multicls(pl_module, batch, test=False):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    cls_logits = infer["multi_modal_cls_feats"]  #  (batch_size, class_num)
    
    cls_labels = batch["labels"]
    
    all_labels = torch.tensor(cls_labels, dtype=torch.long).to(cls_logits.device)
    
    cls_loss = F.cross_entropy(cls_logits, all_labels)

    pred_probs = F.softmax(cls_logits, dim=1)
    
    ret = {
        "multicls_loss": cls_loss,
        "multicls_logits": cls_logits,
        "multicls_labels": all_labels,
        "multicls_pred_probs": pred_probs,
    }
    
    phase = "test" if test else ("train" if pl_module.training else "val")
    

    acc = getattr(pl_module, f"{phase}_multicls_accuracy")(pred_probs, all_labels)
    loss = getattr(pl_module, f"{phase}_multicls_loss")(ret["multicls_loss"])
    

    auroc = getattr(pl_module, f"{phase}_multicls_auroc")(pred_probs, all_labels)

    pl_module.log(f"multicls/{phase}/loss", loss)
    pl_module.log(f"multicls/{phase}/acc", acc)
    pl_module.log(f"multicls/{phase}/auroc", auroc)
    
    return ret
