import torch
import numpy as np
from cardiac_clip.modules import CardiacCLIPTransformer
import os
from transformers import BertTokenizerFast
import SimpleITK as sitk


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device('cuda', 3)

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
    "tokenizer" : "/home/zheng_ying/M3AE-master/huggingface_model/PubMedbert/",
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
    "vit" : '/home/zheng_ying/M3AE-master/huggingface_model/clip/ViT-B-32.pt',
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

def normalization(data):
    _range = np.max(data) - np.min(data)
    print('max',np.max(data))
    print('min',np.min(data))
    return (data - np.min(data)) / _range

def textEncoder(text):
    tokenizer = BertTokenizerFast.from_pretrained(_config["tokenizer"], do_lower_case="uncased" in _config["tokenizer"])
    encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=192,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
    return encoding


def clip_text(pm1,pm2):

    tokens=[textEncoder(pm1),textEncoder(pm2)]
    text_ids=torch.zeros(2,192,dtype=torch.int)
    text_mask=torch.zeros(2,192,dtype=torch.int)

    for i,token in enumerate(tokens):
        input_ids=token["input_ids"]
        attention_mask=token["attention_mask"]
        text_ids[i][:len(input_ids)]=torch.tensor(input_ids)
        text_mask[i][:len(attention_mask)]=torch.tensor(attention_mask)

    text_ids=text_ids.to(device)
    text_mask=text_mask.to(device)

    return text_ids,text_mask

def test_clip(model, image, text_ids, text_masks):
    logit_scale=model.vision_encoder.logit_scale

    #image encode
    image_features = model.vision_encoder(image)
    image_features = model.multi_modal_vision_proj(image_features)

    #text encode
    text_features = model.language_encoder.embeddings(input_ids=text_ids)
    text_input_shape = text_masks.size()
    extended_text_masks = model.language_encoder.get_extended_attention_mask(text_masks, text_input_shape, device)
    for layer in model.language_encoder.encoder.layer:
        text_features = layer(text_features, extended_text_masks)[0]

    text_features=text_features[torch.arange(text_features.shape[0]), text_ids.argmax(dim=-1)]
    text_features = model.multi_modal_language_proj(text_features)

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)


    # cosine similarity as logits
    logit_scale = logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text

def test_zeroshot(model):
    pm_list=[
        {
            'name':'plaque',
            'pm1':"There is calcified plaque at the coronary artery.",
            'pm2':"There is no calcified plaque at the coronary artery."
        },
        {
            'name':'stenosis',
            'pm1':"There is stenosis at the coronary artery.",
            'pm2':"There is no stenosis at the coronary artery."    
        },
        {
            'name':'ao-cal',
            'pm1':'There is calcification at the cardiac aortic wall.',
            'pm2':'There is no calcification at the cardiac aortic wall.'
        },
        {
            'name':'ath',
            'pm1':'There is atherosclerosis.',
            'pm2':'There is no atherosclerosis.'
        },
        {
            'name':'heartshape',
            'pm1':'There is cardiomegaly.',
            'pm2':'There is no cardiomegaly.'
        },
        {
            'name':'effusion',
            'pm1':'There is pericardial effusion.',
            'pm2':'There is no pericardial effusion.'
        },
        {
            'name':'ph',
            'pm1':'There is pulmonary hypertension.',
            'pm2':'There is no pulmonary hypertension.'
        },    
    ]

    with torch.no_grad():
        nii_path='sample_CTA.nii.gz'
        #nii_path='sample_CT.nii.gz'

        nii_image = sitk.ReadImage(nii_path)
        img_npy = sitk.GetArrayFromImage(nii_image)

        print('shape',img_npy.shape) #(128,144,144)
        
        image = (torch.from_numpy(normalization(img_npy))).unsqueeze(0).unsqueeze(0).to(device)
        
        for item in pm_list: 
            name=item['name']
            pm1=item['pm1']
            pm2=item['pm2']

            text_ids, text_mask=clip_text(pm1,pm2)
            logits_per_image, logits_per_text = test_clip(model, image, text_ids, text_mask)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            res='Yes'if probs[0][0]>probs[0][1] else 'No'
            
            print('---------------------------------------')
            print("Name:",name)
            print("Probs:", probs) 
            print('Prediction:',res)



check_model = torch.load('cardiac_clip.ckpt',map_location='cpu')

checkpoint = check_model['state_dict']

model=CardiacCLIPTransformer(_config)
model.load_state_dict(checkpoint)

model.eval()
model.to(device)

test_zeroshot(model)