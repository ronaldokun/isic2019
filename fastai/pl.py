# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] id="tkQEDQA44qlC"
# # Melanoma classification with PyTorch Lightning
#
# Using EfficientNet on PyTorch Lightning, with its amazing hardware agnostic and mixed precision implementation.
#
# This is still work in progress, so please bear with me

# + id="ftZsAlKR4qlF"
from pathlib import Path
from fastprogress import progress_bar as tqdm
fold_number = 1
seed  = 66
debug = False
tta   = 20

batch_size = {
    'tpu': 10, # x8
    'gpu': 16, # 10 without AMP
    'cpu': 4,
}

arch = 'efficientnet-b5'
resolution = 456  # orignal res for B5
input_res  = 512

lr = 8e-6   # * batch_size
weight_decay = 2e-5
pos_weight   = 3.2
label_smoothing = 0.03

max_epochs = 7

from data import DATA, TRAIN, TEST, OUT
DATA_PATH = DATA
TRAIN_ROOT_PATH = TRAIN
TEST_ROOT_PATH = TEST
SAVE_DIR = OUT / f'pl/fold_{fold_number}'
SAVE_DIR.mkdir(exist_ok=True, parents=True)
# -


# #!pip install --upgrade wandb
# !wandb login 6ff8d5e5bd920e68d1f76b574f1880278b4ac8d2

# + [markdown] id="UnEoyD1a4qlO"
# # Why PyTorch Lightning?
# Lightning is simply organized PyTorch code. There's NO new framework to learn.
# For more details about Lightning visit the repo:
#
# https://github.com/PyTorchLightning/pytorch-lightning
#
# - Run on CPU, GPU clusters or TPU, without any code changes
# - Transparent use of AMP (automatic mixed precision)
#
# ![lightning structure](https://raw.githubusercontent.com/PyTorchLightning/pytorch-lightning/master/docs/source/_images/lightning_module/pt_to_pl.png)

# + [markdown] id="oa7QSur04qlR"
# # Install modules
#
# Update PyTorch to enable its native support to Mixed Precision or XLA for TPU

# + _kg_hide-input=false _kg_hide-output=true id="_lmyKFye4qlT" outputId="a008eca4-4ec3-4ac7-a1a8-7f72cfa3c337"
# %reload_ext autoreload
# %autoreload 2
import os

if 'TPU_NAME' in os.environ.keys():
    try:
        import torch_xla
    except:
        pass
        # XLA powers the TPU support for PyTorch
        # !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
        # !python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
else:
    pass
    # Update PyTorch to enable its native support to Mixed Precision
    # !pip install --pre torch==1.7.0.dev20200701+cu101 torchvision==0.8.0.dev20200701+cu101 -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html

# #!pip3 install -U pip albumentations==0.4.5 PyYAML pytorch-lightning==0.8.5 efficientnet_pytorch
# -

import wandb
from pytorch_lightning.loggers import WandbLogger

# + [markdown] id="NRdfuFiR4qlf"
# # Hardware lookup

# + _kg_hide-input=true id="t9FS-xGM4qlh" outputId="5802bc07-8b6e-4254-ac49-2139d11911f8"
import os
import torch

num_workers = os.cpu_count()
gpus = 1 if torch.cuda.is_available() else None

print(f'GPUs: {gpus}')

try:
    import torch_xla.core.xla_model as xm
    tpu_cores = 8 #xm.xrt_world_size()
except:
    tpu_cores = None

# + _kg_hide-input=true id="t9FS-xGM4qlh" outputId="5802bc07-8b6e-4254-ac49-2139d11911f8"
if isinstance(batch_size, dict):
    if tpu_cores:
        batch_size = batch_size['tpu']
        lr *= tpu_cores
        num_workers = 1
    elif gpus:
        batch_size = batch_size['gpu']
        # support for free Colab GPU's
        if 'K80' in torch.cuda.get_device_name():
            batch_size = batch_size//3
        elif 'T4' in torch.cuda.get_device_name():
            batch_size = int(batch_size * 0.66)
    else:
        batch_size = batch_size['cpu']

lr *= batch_size

dict(
    num_workers=num_workers,
    tpu_cores=tpu_cores,
    gpus=gpus,
    batch_size=batch_size,
    lr=lr,
)

# + [markdown] id="fa--zHZ64qln"
# # Automatic Mixed Precision
#
# NVIDIA Apex is required only prior to PyTorch 1.6

# + _kg_hide-input=true _kg_hide-output=true id="2jNntNJb4qln"
# check for torch's native mixed precision support (pt1.6+)
if gpus and not hasattr(torch.cuda, "amp"):
    try:
        from apex import amp
    except:
        # !git clone https://github.com/NVIDIA/apex  nv_apex
        # !pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./nv_apex
        from apex import amp
    # with PyTorch Lightning all you need to do now is set precision=16

# + [markdown] id="SaoinaAu4qlu"
# # Imports

# + _kg_hide-input=true _kg_hide-output=true id="UFXXIDJt4qlv" outputId="b5e99e4c-016c-440c-e2f4-c3ed45ba50eb"
import os
import time
import random
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
from skimage import io
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from glob import glob
import sklearn

import pytorch_lightning as pl
import warnings

warnings.filterwarnings("ignore") 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(seed*6 + fold_number)

torch.__version__

# + [markdown] id="oeM18dHg4qlz"
# # Dataset
#
# We will be using @shonenkov dataset with external data: https://www.kaggle.com/shonenkov/melanoma-merged-external-data-512x512-jpeg 
#
# thank you @shonenkov

# + _kg_hide-input=true id="MQ0g7uTd4ql0"
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, path, image_ids, labels=None, transforms=None):
        super().__init__()
        self.path = path
        self.image_ids = image_ids
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image = cv2.imread(f'{self.path}/{image_id}.jpg', cv2.IMREAD_COLOR)

        if self.transforms:
            sample = self.transforms(image=image)
            image  = sample['image']

        label = self.labels[idx] if self.labels is not None else 0.5
        return image, label

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def get_labels(self):
        return list(self.labels)


# + [markdown] id="vlBehFHZ4ql4"
# # Augmentations

# + _kg_hide-input=false id="9We_dtmL4ql6"
def get_train_transforms():
    return A.Compose([
            A.JpegCompression(p=0.5),
            A.Rotate(limit=80, p=1.0),
            A.OneOf([
                A.OpticalDistortion(),
                A.GridDistortion(),
                A.IAAPiecewiseAffine(),
            ]),
            A.RandomSizedCrop(min_max_height=(int(resolution*0.7), input_res),
                              height=resolution, width=resolution, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussianBlur(p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(),   
                A.HueSaturationValue(),
            ]),
            A.Cutout(num_holes=8, max_h_size=resolution//8, max_w_size=resolution//8, fill_value=0, p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ], p=1.0)

def get_valid_transforms():
    return A.Compose([
            A.CenterCrop(height=resolution, width=resolution, p=1.0),
            A.Normalize(),
            ToTensorV2(),
        ], p=1.0)

def get_tta_transforms():
    return A.Compose([
            A.JpegCompression(p=0.5),
            A.RandomSizedCrop(min_max_height=(int(resolution*0.9), int(resolution*1.1)),
                              height=resolution, width=resolution, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ], p=1.0)


# +
import numpy as np
import random
import pandas as pd
from collections import Counter, defaultdict

def stratified_group_k_fold(X, y, groups, k, seed=None):
    """ https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in tqdm(sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])), total=len(groups_and_y_counts)):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


# +
# df_folds = pd.read_csv(f'{DATA_PATH}/external_upsampled_tabular.csv').rename({'image_name': 'image_id'}, axis=1)

# df2 = pd.read_csv(f'{DATA_PATH}/folds_13062020.csv')

# df_folds = pd.merge(df_folds, df2, on=['image_id'], how='left').iloc[:, 0:8]
# df_folds.columns = ['image_id', 'sex', 'age_approx', 'anatom_site_general_challenge', 'target', 'width', 'height', 'patient_id']

# df_folds['patient_id'] = df_folds['patient_id'].fillna(df_folds['image_id'])
# df_folds['sex'] = df_folds['sex'].fillna('unknown')
# df_folds['anatom_site_general_challenge'] = df_folds['anatom_site_general_challenge'].fillna('unknown')
# df_folds['age_approx'] = df_folds['age_approx'].fillna(round(df_folds['age_approx'].mean()))
# patient_id_2_count = df_folds[['patient_id', 'image_id']].groupby('patient_id').count()['image_id'].to_dict()

# df_folds = df_folds.set_index('image_id')

# def get_stratify_group(row):
#     stratify_group = row['sex']
# #     stratify_group += f'_{row["anatom_site_general_challenge"]}'
#     stratify_group += f'_{row["target"]}'
#     patient_id_count = patient_id_2_count[row["patient_id"]]
#     if patient_id_count > 80:
#         stratify_group += f'_80'
#     elif patient_id_count > 60:
#         stratify_group += f'_60'
#     elif patient_id_count > 50:
#         stratify_group += f'_50'
#     elif patient_id_count > 30:
#         stratify_group += f'_30'
#     elif patient_id_count > 20:
#         stratify_group += f'_20'
#     elif patient_id_count > 10:
#         stratify_group += f'_10'
#     else:
#         stratify_group += f'_0'
#     return stratify_group

# df_folds['stratify_group'] = df_folds.apply(get_stratify_group, axis=1)
# df_folds['stratify_group'] = df_folds['stratify_group'].astype('category').cat.codes

# df_folds.loc[:, 'fold'] = 0

# skf = stratified_group_k_fold(X=df_folds.index, y=df_folds['stratify_group'], groups=df_folds['patient_id'], k=5, seed=42)

# for fold_number, (train_index, val_index) in enumerate(skf):
#     df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

# df_folds.to_csv(DATA / 'upsample.csv')

# + [markdown] id="6K-p9nPl4ql-"
# # Setup dataset

# + id="Dq3S8RAF4ql_" outputId="00122522-4e2b-4e79-adcb-6501fea3ff78"
df_folds = pd.read_csv(f'{DATA_PATH}/upsample.csv', index_col='image_id',
                       usecols=['image_id', 'fold', 'target'], dtype={'fold': np.byte, 'target': np.byte})

_ = df_folds.groupby('fold').target.hist(alpha=0.4)
df_folds.groupby('fold').target.mean().to_frame('ratio').T

# + id="3h2Zcdod4qmE"
df_test = pd.read_csv(f'{DATA}/test.csv', index_col='image_name')

if debug:
    df_folds = df_folds.sample(batch_size * 80)

df_folds = df_folds.sample(frac=1.0, random_state=seed*6+fold_number)

# + id="5i8UhuvM4qmJ" outputId="c0ea1e24-01ca-4660-af87-df919cdc884c"
ds_train = ImageDataset(
    path=TRAIN_ROOT_PATH,
    image_ids=df_folds[df_folds['fold'] != fold_number].index.values,
    labels=df_folds[df_folds['fold'] != fold_number].target.values,
    transforms=get_train_transforms(),
)

ds_val = ImageDataset(
    path=TRAIN_ROOT_PATH,
    image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
    labels=df_folds[df_folds['fold'] == fold_number].target.values,
    transforms=get_valid_transforms(),
)

ds_test = ImageDataset(
    path=TEST_ROOT_PATH,
    image_ids=df_test.index.values,
    transforms=get_tta_transforms(),
)

del df_folds
len(ds_train), len(ds_val), len(ds_test)

# + [markdown] id="dqfK4rxh4qmO"
# # Model

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" id="vMgS0GCj4qmP"
from efficientnet_pytorch import EfficientNet
from pytorch_lightning.metrics.classification import AUROC
from sklearn.metrics import roc_auc_score

class Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = EfficientNet.from_pretrained(arch, advprop=True)
        self.net._fc = nn.Linear(in_features=self.net._fc.in_features, out_features=1, bias=True)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            max_lr=lr,
            epochs=max_epochs,
            optimizer=optimizer,
            steps_per_epoch=int(len(ds_train) / batch_size),
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100,
            base_momentum=0.90,
            max_momentum=0.95,
        )
        return [optimizer], [scheduler]

    def step(self, batch):
        # return batch loss
        x, y  = batch
        y_hat = self(x).flatten()
        y_smo = y.float() * (1 - label_smoothing) + 0.5 * label_smoothing
        loss  = F.binary_cross_entropy_with_logits(y_hat, y_smo.type_as(y_hat),
                                                   pos_weight=torch.tensor(pos_weight))
        return loss, y, y_hat.sigmoid()

    def training_step(self, batch, batch_nb):
        # hardware agnostic training
        loss, y, y_hat = self.step(batch)
        acc = (y_hat.round() == y).float().mean().item()
        tensorboard_logs = {'train_loss': loss, 'acc': acc}
        return {'loss': loss, 'acc': acc, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss, y, y_hat = self.step(batch)
        return {'val_loss': loss,
                'y': y.detach(), 'y_hat': y_hat.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        auc = AUROC()(pred=y_hat, target=y) if y.float().mean() > 0 else 0.5 # skip sanity check
        acc = (y_hat.round() == y).float().mean().item()
        print(f"Epoch {self.current_epoch} acc:{acc} auc:{auc}")
        tensorboard_logs = {'val_loss': avg_loss, 'val_auc': auc, 'val_acc': acc}
        return {'avg_val_loss': avg_loss,
                'val_auc': auc, 'val_acc': acc,
                'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, _ = batch
        y_hat = self(x).flatten().sigmoid()
        return {'y_hat': y_hat}

    def test_epoch_end(self, outputs):
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        assert len(df_test) == len(y_hat), f"{len(df_test)} != {len(y_hat)}"
        df_test['target'] = y_hat.tolist()
        N = len(glob('submission*.csv'))
        df_test.target.to_csv(f'submission{N}.csv')
        return {'tta': N}

    def train_dataloader(self):
        return DataLoader(ds_train, batch_size=batch_size, num_workers=num_workers,
                          drop_last=True, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers,
                          drop_last=False, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(ds_test, batch_size=batch_size, num_workers=num_workers,
                          drop_last=False, shuffle=False, pin_memory=False)


# -

checkpoints = sorted(list(SAVE_DIR.iterdir()), key=lambda x: int(x.stem.split('_')[0]))
#checkpoints = sorted([p for p in SAVE_DIR.iterdir()], key=lambda x: float(x.stem.split("=")[-1]))
if len(checkpoints):
    checkpoint = str(checkpoints[0])
else:
    checkpoint=None
#+ _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" id="vMgS0GCj4qmP"
model = Model()#.load_from_checkpoint(str(checkpoint))
# -

wandb.init(project='melanoma', tags=['amp', 'lightning'], name='upsampled_full_data')
wandb_logger = WandbLogger(project='melanoma', tags=['amp', 'lightning'], name='upsampled_full_data')
wandb.watch(model)

# +
# Plot some training images
#import torchvision.utils as vutils
#batch, targets = next(iter(model.train_dataloader()))

#plt.figure(figsize=(16, 8))
#plt.axis("off")
#plt.title("Training Images")
#_ = plt.imshow(vutils.make_grid(
#    batch[:16], nrow=8, padding=2, normalize=True).cpu().numpy().transpose((1, 2, 0)))

#targets[:16].reshape([2, 8]) if len(targets) >= 16 else targets

# + _kg_hide-input=true
# # test the same images
# with torch.no_grad():
#     print(model(batch[:16]).reshape([len(targets)//8,8]).sigmoid())
#del batch; del targets

# + [markdown] id="_1ib_IBN4qmS"
# # Train
# The Trainer automates the rest.
#
# Trains on 8 TPU cores, GPU or CPU - whatever is available.

# +
# # View logs life in tensorboard
# Unfortunately broken again in the Kaggle notebooks :(
# however, it still works nicely in Colab or locally :)

# if gpus:
# #     !pip install -qU tensorboard-plugin-profile
# # %reload_ext tensorboard
# # %tensorboard --logdir lightning_logs/

# + _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" id="4AEUW-iB4qmT" outputId="49468221-86ef-4e7b-b627-7d51397fad08"
checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath="{epoch:02d}_{val_auc:.4f}",save_top_k=1, monitor='val_auc', mode='max')
if len(checkpoints):
    trainer = pl.Trainer(resume_from_checkpoint=checkpoint,
                         default_root_dir=SAVE_DIR,
                         gpus=gpus, 
                         precision=16 if gpus else 32,
                         max_epochs=max_epochs,
                         checkpoint_callback=checkpoint_callback,
                         logger=wandb_logger,
                         num_sanity_val_steps=1)
else:
    trainer = pl.Trainer(
        default_root_dir=SAVE_DIR,
        tpu_cores=tpu_cores,
        gpus=gpus,
        precision=16 if gpus else 32,
        max_epochs=max_epochs,
        num_sanity_val_steps=1 if debug else 0,
        checkpoint_callback=checkpoint_callback,
        logger=wandb_logger
    #     val_check_interval=0.25, # check validation 4 times per epoch
        )

# + _kg_hide-input=true _kg_hide-output=true
# clean up gpu in case you are debugging 
import gc
torch.cuda.empty_cache(); gc.collect()
torch.cuda.empty_cache(); gc.collect()

# + id="GpASvAkY4qmb" outputId="ed7fc27d-3b35-4ca9-d479-b2318582d38c"
trainer.fit(model)


