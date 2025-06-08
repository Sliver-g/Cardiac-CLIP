import torch
import torch.nn as nn
from transformers import get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers.optimization import AdamW
from ..gadgets.my_metrics import Accuracy, Scalar, ROCScore,MultiClassROCScore


def set_metrics(pl_module):
    for split in ["train", "val","test"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v <= 0:
                continue

            if k == "cls":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_auroc", ROCScore())

            elif k == "multicls":
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                    setattr(pl_module, f"train_{k}_auroc", MultiClassROCScore(num_classes=4))
                else:
                    setattr(pl_module, f"val_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"val_{k}_loss", Scalar())
                    setattr(pl_module, f"val_{k}_auroc",  MultiClassROCScore(num_classes=4))
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_auroc",  MultiClassROCScore(num_classes=4))

            elif k == "irtr":
                if split == "train":
                    setattr(pl_module, f"train_irtr_loss", Scalar())
                else:
                    setattr(pl_module, f"val_irtr_loss", Scalar())
                    setattr(pl_module, f"test_irtr_loss", Scalar())

            elif k == "itm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

            elif k == "mlm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

            elif k == "mim":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

            elif k == "itc":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

            else:
                raise ValueError


def epoch_wrapup(pl_module, test=False):
    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    the_metric = 0

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v <= 0:
            continue
        value = 0

        if loss_name == "multicls":
            # value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            # pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            # getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            value = getattr(pl_module, f"{phase}_{loss_name}_auroc").compute()
            pl_module.log(f"{loss_name}/{phase}/auroc_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_auroc").reset()

            pl_module.log(f"{loss_name}/{phase}/loss_epoch", getattr(pl_module, f"{phase}_{loss_name}_loss").compute())
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        
        elif loss_name == "cls":
            value = getattr(pl_module, f"{phase}_{loss_name}_auroc").compute()
            pl_module.log(f"{loss_name}/{phase}/auroc_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_auroc").reset()

            pl_module.log(f"{loss_name}/{phase}/loss_epoch", getattr(pl_module, f"{phase}_{loss_name}_loss").compute())
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        elif loss_name == "mim":
            value = -getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            pl_module.log(f"{loss_name}/{phase}/loss_epoch", - value)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        elif loss_name == "itc":
            value = -getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            pl_module.log(f"{loss_name}/{phase}/loss_epoch", - value)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
        else:
            raise ValueError

        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric)


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [k for k, v in pl_module.hparams.config["loss_names"].items() if v > 0]
    return


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]
    lr_multiplier_head = pl_module.hparams.config["lr_multiplier_head"]
    lr_multiplier_multi_modal = pl_module.hparams.config["lr_multiplier_multi_modal"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["mim_head", 'cls_head']  
    multi_modal_names = ['multi_modal']

    layers_to_freeze=[]
    freeze=[]
     # freeze layer
    for layer in layers_to_freeze:
        for name,param in layer.named_parameters():
            print(name)
            if(name!=''and name!='visual' and name!='visual.transformer' and name!='visual.transformer.resblocks'):
              #and'resblocks.9'not in name and'resblocks.10'not in name and'layer.10'not in name and"pooler"not in name
              #and'layer.11'not in name and'ln_post'not in name):
                freeze.append('language_encoder.'+name)
                param.requires_grad = False
            # if('conv1'in name):
            #     freeze.append(name)
            #     param.requires_grad = False
    #freeze=[]
    for n, p in pl_module.named_parameters():
        print(n)

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                   and not any(bb in n for bb in head_names)
                   and not any(ht in n for ht in multi_modal_names)
                   and not any(fz in n for fz in freeze)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                   and not any(bb in n for bb in head_names)
                   and not any(ht in n for ht in multi_modal_names)
                   and not any(fz in n for fz in freeze)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                   and any(bb in n for bb in head_names)
                   and not any(ht in n for ht in multi_modal_names)
                   and not any(fz in n for fz in freeze)
            ],
            "weight_decay": wd,
            "lr": lr * lr_multiplier_head,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                   and not any(ht in n for ht in multi_modal_names)
                   and not any(fz in n for fz in freeze)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_multiplier_head,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                   and not any(bb in n for bb in head_names)
                   and any(ht in n for ht in multi_modal_names)
                   and not any(fz in n for fz in freeze)
            ],
            "weight_decay": wd,
            "lr": lr * lr_multiplier_multi_modal,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                   and not any(bb in n for bb in head_names)
                   and any(ht in n for ht in multi_modal_names)
                   and not any(fz in n for fz in freeze)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_multiplier_multi_modal,
        },
    ]


    if optim_type == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    else:
        raise ValueError

    if pl_module.trainer.max_steps is None:
        max_steps = (
                len(pl_module.trainer.datamodule.train_dataloader())
                * pl_module.trainer.max_epochs
                // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return [optimizer], [sched]
