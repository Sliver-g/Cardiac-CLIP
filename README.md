# Cardiac-CLIP
<img src="https://github.com/user-attachments/assets/324fa6c0-5c26-4a66-9e88-7d7a691c5d6b" alt="Cardiac-CLIP" >

# Requirements
Before running the code, you need to install the required dependencies. You can install all the necessary packages by running the following command:

```bash
pip install -r cardiac_clip_requirements.txt
```

# Training
The training and fine-tuning code is located in [train.py](Cardiac_CLIP/train.py). You can switch between pre-training and fine-tuning by configuring the loss_name parameter.

'itc' indicates pre-training using contrastive loss.

'cls' and 'multi_cls' indicate fine-tuning for single-class and multi-class classification, respectively.

# Inference
The inference code is in [test_zeroshot.py](Cardiac_CLIP/test_zeroshot.py), which performs zero-shot inference by loading the pre-trained weights of Cardiac-CLIP. The example prompts include seven different types of heart diseases. Sample CTA and CT images are provided in the Sample folder. The labels and inference results are as follows:

<img src="https://github.com/user-attachments/assets/9f1ccd2d-6e48-48c3-b626-80a05dcacad7" alt="Sample_CTA">
<img src="https://github.com/user-attachments/assets/b8532d2d-249b-44fd-989b-acead8093f12" alt="Sample_CT" >


# Pretrained Model
You can download the pre-trained parameters of Cardiac-CLIP from [this link](https://drive.google.com/file/d/1fxvOIYb_xMHYmTlHAvDmcMBPdPn6-uCn/view?usp=drive_link).

# Citing Us
If you use Cardiac-CLIP in your research, we appreciate your citation of our paper, which can be found [here].
