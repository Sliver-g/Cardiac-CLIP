import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from transformers.models.bert.modeling_bert import BertConfig, BertModel

from cardiac_clip.modules import cardiac_clip_utils
from cardiac_clip.modules import objectives
from cardiac_clip.modules import prediction_heads
from cardiac_clip.modules.language_encoders.bert_model import BertCrossLayer
from cardiac_clip.modules.cardiac_clip_utils import init_weights
from cardiac_clip.modules.vision_encoders.clip_model import build_model, adapt_position_encoding

from timm.models.layers import trunc_normal_
#from pytorch_lightning.utilities.distributed import DistributedSamplerWrapper


class CardiacCLIPTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # == Begin: 1. Build Models ==
        self.is_clip = ('swin' not in config['vit'])
        if 'bert' in config['tokenizer']:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            raise ValueError

        resolution_after = config['image_size']
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(config['vit'], resolution_after=resolution_after)
            
                BertModel.from_pretrained(config['tokenizer'])
            torch.distributed.barrier()
        if self.is_clip:
            self.vision_encoder = build_model(config['vit'], resolution_after=resolution_after)
        self.language_encoder = BertModel.from_pretrained(config['tokenizer'])

        self.multi_modal_language_proj = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.multi_modal_language_proj.apply(init_weights)
        self.multi_modal_vision_proj = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.multi_modal_vision_proj.apply(init_weights)

        # == End  : 1. Build Models ==

        # == Begin: 2. Build Pre-Training Heads ==
        if config["loss_names"]["mim"] > 0:
            self.mim_head = prediction_heads.MIMHead(config)
            self.mim_head.apply(init_weights)
        # == End  : 2. Build Pre-Training Heads ==

        # == Begin: 3. Load Models ==
        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                # state_dict = adapt_position_encoding(state_dict,
                #                                      after=resolution_after,
                #                                      patch_size=self.hparams.config['patch_size'])
                pass

            self.load_state_dict(state_dict, strict=True)
        # == End  : 3. Load Models ==

        # == 4. Build Heads For Downstream Tasks ==
        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["cls"] > 0 or self.hparams.config["loss_names"]["multicls"] > 0:
            ms= 1
            self.cls_head=nn.Sequential(
                nn.Linear(hs, hs),
                nn.LayerNorm(hs),
                nn.GELU(),
                nn.Linear(hs, ms),
            )
            self.cls_head.apply(init_weights)

        cardiac_clip_utils.set_metrics(self)
        self.current_tasks = list()
        # == End:  4. Build Heads For Downstream Tasks ==

        # == Begin: 5. Load Models For Testing ==
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            # state_dict = adapt_position_encoding(state_dict, after=resolution_after,
            #                                      patch_size=self.hparams.config['patch_size'])
            self.load_state_dict(state_dict, strict=False)
        # == End  : 5. Load Models For Testing ==

    def random_masking(self, x, mask_ratio):
        x_ = x[:, :1]
        x = x[:, 1:]
        pos_embed = self.vision_encoder.visual.positional_embedding.unsqueeze(0).to(x)  #(1,513,512)

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x += pos_embed[:, 1:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # append cls token
        x_ = x_ + pos_embed[:, :1]
        x_masked = torch.cat((x_, x_masked), dim=1)

        return x_masked, mask, ids_restore

    def patchify(self, imgs):
        p = self.hparams.config["patch_size"]
        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        #h = w = z = imgs.shape[2] // p
        h=imgs.shape[2]//p
        w=imgs.shape[3]//p
        z=imgs.shape[4]//p
        #x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, z, p))
        #x = torch.einsum('nchpwq->nhwpqc', x)
        x = torch.einsum('nchpwqzr->nhwzpqrc', x)
        #x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        x = x.reshape(shape=(imgs.shape[0], h * w * z, p ** 3 * 1))
        return x

    def unpatchify(self, x):
        p = self.hparams.config["patch_size"]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            img=None,
            output_attentions=False,
            unimodal=False
    ):
        ret = dict()

        # == Begin: Fetch the inputs ==
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                img_key = f"image_{image_token_type_idx - 1}"
            else:
                img_key = "image"
            img = batch[img_key]
            
        text_ids = batch[f"text_ids"]
        text_masks = batch[f"text_masks"]
        device = text_ids.device
        device=img.device

        # # == Begin: Text Encoding ==
        uni_modal_text_feats = self.language_encoder.embeddings(input_ids=text_ids)
        text_input_shape = text_masks.size()
        extended_text_masks = self.language_encoder.get_extended_attention_mask(text_masks, text_input_shape, device)
        for layer in self.language_encoder.encoder.layer:
            uni_modal_text_feats = layer(uni_modal_text_feats, extended_text_masks)[0]

        uni_modal_text_feats=uni_modal_text_feats[torch.arange(uni_modal_text_feats.shape[0]), text_ids.argmax(dim=-1)]
        uni_modal_text_feats = self.multi_modal_language_proj(uni_modal_text_feats)
        # == End  : Text Encoding ==

        # == Begin: Image Encoding ==
        uni_modal_image_feats = self.vision_encoder(img)  #(1,1,512)

        #cls_feats =self.fc_norm(uni_modal_image_feats[:,1:,:])
        uni_modal_image_feats = self.multi_modal_vision_proj(uni_modal_image_feats)
        #cls_feats=self.cls_head(uni_modal_image_feats)

        #uni_modal_image_feats = torch.from_numpy(np.mean(uni_modal_image_feats.cpu().detach().numpy(), axis=1)).to(device)
        

        ret.update({
            "images": img,
            "patched_images": self.patchify(img),
            # "text_labels": text_labels,
            # "text_ids": text_ids,
            # "text_masks": text_masks,
            # #"extended_image_masks": extended_image_masks,
            # "extended_text_masks": extended_text_masks,
            "multi_modal_text_feats": uni_modal_text_feats,
            "multi_modal_image_feats": uni_modal_image_feats,
            #"multi_modal_cls_feats": cls_feats,
        })

        return ret

    def forward(self, batch, test=False):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret


        # Pre-Training: Masked Image Modeling
        if "mim" in self.current_tasks:
            ret.update(objectives.compute_mim(self, batch))

        
        # Pre-Training: CLIP
        if "itc" in self.current_tasks:
            ret.update(objectives.compute_itc(self, batch))

        # Fine-Tuning: Image-Text Classification
        if "cls" in self.current_tasks:
            ret.update(objectives.compute_cls(self, batch, test=test))
        
        if "multicls" in self.current_tasks:
            ret.update(objectives.compute_multicls(self, batch, test=test))


        return ret

    def on_train_start(self) -> None:
        print('Training started')

    # def on_train_epoch_start(self):
    #     #print("epoch start!")
    #     
    #     if self.current_epoch <= 2:
    #         for param in self.vision_encoder.parameters():
    #             param.requires_grad = False  
    #     else:
    #         for param in self.vision_encoder.parameters():
    #             param.requires_grad = True  
    #     self.vision_encoder.eval() if self.current_epoch <=2 else self.vision_encoder.train()


    def training_step(self, batch, batch_idx):
        cardiac_clip_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v * self.hparams.config["loss_names"][k.replace("_loss", "")]
                          for k, v in output.items() if "loss" in k])

        self.log('loss',total_loss,prog_bar=True)

        return total_loss


    # def training_epoch_end(self, outs):
    #     m3ae_utils.epoch_wrapup(self)

    def on_train_epoch_end(self):
        cardiac_clip_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        cardiac_clip_utils.set_task(self)
        output = self(batch)

    def on_validation_epoch_end(self):
        cardiac_clip_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        cardiac_clip_utils.set_task(self)
        output = self(batch, test=True)

    # def test_epoch_end(self, outs):
    #     m3ae_utils.epoch_wrapup(self, test=True)

    def on_test_epoch_end(self):
        cardiac_clip_utils.epoch_wrapup(self, test=True)

    def configure_optimizers(self):
        return cardiac_clip_utils.set_schedule(self)
    

