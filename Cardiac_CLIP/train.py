import copy
import os
import resource

import pytorch_lightning as pl
#from pytorch_lightning.callbacks import TQDMProgressBar

from cardiac_clip.datamodules.multitask_datamodule import MTDataModule
from cardiac_clip.modules import CardiacCLIPTransformer
import torch
import warnings

warnings.filterwarnings("ignore")

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# class MeterlessProgressBar(TQDMProgressBar):
#     def get_metrics(self, trainer, model):
#         # don't show the version number
#         items = super().get_metrics(trainer, model)
#         items.pop("v_num", None)
#         return items

def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # Data modules
    dm = MTDataModule(_config, dist=True)

    # Module
    model = CardiacCLIPTransformer(_config)

    # Loggers
    os.makedirs(_config["log_dir"], exist_ok=True)
    exp_name = f'{_config["exp_name"]}'
    run_name = f'{exp_name}-seed{_config["seed"]}-from_{_config["load_path"].replace("/", "_")}'
    tb_logger = pl.loggers.TensorBoardLogger(_config["log_dir"], name=run_name)

    loggers = [tb_logger]

    # Callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
        save_weights_only=True if "finetune" in exp_name else False,
        #every_n_train_steps=200
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    # Training Hyper-Parameters
    #num_gpus = (_config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"]))
    # grad_steps = max(_config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]), 1)
    grad_steps = 1
    max_steps = _config["max_steps"]
    max_epochs = _config["max_epoch"]

    # Trainer
    trainer = pl.Trainer(
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="gpu",
        strategy = "ddp_find_unused_parameters_true",
        benchmark=True,
        deterministic=True,
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=loggers,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        #num_sanity_val_steps=0,
        default_root_dir=_config["default_root_dir"],
        devices=[3],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
        if "finetune" in exp_name:
            trainer.test(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)

_loss_names = {
    "mim": 0,
    "itc": 1,
    "cls": 0,
    "multicls":0,
}

_config = {
    "exp_name" : "train_cardiac_clip", 
    "datasets":["Jinling"],
    "loss_names" : _loss_names,
    "batch_size" : 16,  
    "max_epoch" : 10,
    "max_steps" :  25970,
    "warmup_steps" : 0.05,
    "whole_word_masking" : True,

    "vocab_size" : 30522,
    "max_text_len" : 160,
    "image_size" : (128,144,144),
    "tokenizer" : "./huggingface_model/PubMedbert/",
    "train_transform_keys" : ["clip"],
    "val_transform_keys" : ["clip"],
    "learning_rate" : 1e-5,

    "lr_multiplier_head" : 5, 
    "lr_multiplier_multi_modal" : 5,
    "num_top_layer" : 6,
    "hidden_size" : 768,
    "num_heads" : 12,

    "precision" : 16,
    "mim_layer" : 3,
    
    "num_gpus" : 2,
    "per_gpu_batchsize" : 16,
    "clip16" : 1,
    "data_root":"Jundata_with_text_lung_and_construct_oneval_nonon_labelno0_v2.json",

    
    "seed": 0,
    "num_workers" : 0,

    # Text Setting
    "vqa_label_size" : 3129,
    "mlc_label_size" : 14,
    "mlm_prob" : 0.15,
    "draw_false_text" : 0,

    # Image setting
    "patch_size" : 8,
    "draw_false_image" : 1,
    "image_only" : False,

    # Transformer Setting
    "input_image_embed_size" : 768,
    "input_text_embed_size" : 768,
    "vit" : './huggingface_model/clip/ViT-B-32.pt',
    "num_layers" : 6,
    "mlp_ratio" : 4,
    "drop_rate" : 0.1,

    # MIM decoder Setting
    "mim_prob" : 0.75,
    "mim_decoder_hidden_size" : 384,
    "mim_decoder_num_layers" : 4,
    "mim_decoder_num_heads" : 6,
    "norm_pix_loss" : True,
    "mim_layer" : -1,

    # Optimizer Setting
    "optim_type" : "adamw",
    "weight_decay" : 0.01,    ###原0.01
    "decay_power" : 1,
    "end_lr" : 0,

    # Downstream Setting
    "get_recall_metric" : False,

    # PL Trainer Setting
    "resume_from" : None,
    "fast_dev_run" : False,
    "val_check_interval" : 1.0,
    "test_only" : False,
    "default_root_dir" : "checkpoints",

    # below params varies with the environment
    "log_dir" : "your_log_dir",
    "num_nodes" : 1,
    "load_path":'',

    # MELINDA SETTING
    "label_column_name" : "",
    "melinda_label_size" : {"i_meth": 85, "p_meth": 45, "i_meth_label": 15, "p_meth_label": 7},
}

if __name__ == '__main__':
    main(_config)
