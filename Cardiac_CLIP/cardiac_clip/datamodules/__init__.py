from .pretraining_jinling_datamodule import JinlingDataModule
from .cls_nlst_datamodule import NLSTDataModule
_datamodules = {
    "Jinling":JinlingDataModule,
    'NLST':NLSTDataModule,
}
