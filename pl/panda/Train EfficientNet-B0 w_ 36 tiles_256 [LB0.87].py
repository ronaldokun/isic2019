import os
from pathlib import Path
import sys
import time
import skimage.io
import numpy as np
import pandas as pd
import cv2
import PIL
from PIL import Image
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler
from radam import *
from csvlogger import *
from mish_activation import *
from efficientnet_pytorch import EfficientNet
import albumentations
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm


# +
def load_image(fn, mode=None, **kwargs):
    "Open and load a `PIL.Image` and convert to `mode`"
    im = PIL.Image.open(fn, **kwargs)
    im.load()
    im = im._new(im.im)
    return im.convert(mode) if mode else im

# Cell
def image2tensor(img):
    "Transform image to byte tensor in `c*h*w` dim order."
    res = tensor(img)
    if res.dim()==2: res = res.unsqueeze(-1)
    return res.permute(2,0,1)

def pil2numpy(image,dtype:np.dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
#     if a.ndim==2 : a = np.expand_dims(a,2)
#     a = np.transpose(a, (1, 0, 2))
#     a = np.transpose(a, (2, 1, 0))
    return a.astype(dtype, copy=False)


# +
def pil2tensor(image,dtype:np.dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False) )

def open_image(fn, div=False, convert_mode='RGB'):
    x = load_image(fn, mode=convert_mode)
    x = pil2numpy(x,np.float32)
    if div: 
        #x.div_(255)
        x = x/255.0
    #return 1.0 - x #invert image for zero padding
    return x


data_dir = Path('/content/clouderizer/panda/data')
TRAIN = '/content/clouderizer/panda/data/train/'
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv')).sort_values('image_id')
#image_folder = os.path.join(data_dir, 'train_images')
tiles_folder = TRAIN

enet_type = 'efficientnet-b0'
fold = 0
tile_size = 128
image_size = 128
n_tiles = 16
batch_size = 32
num_workers = 4
out_dim = 5
init_lr = 1e-2
warmup_factor = 10
warmup_epo = 1
n_epochs = 30

wandb.init(name='panda-competition',
           config=dict(enet_type = 'efficientnet-b0',
                fold = 0,
                tile_size = 128,
                image_size = 128,
                n_tiles = 16,
                batch_size = 32,
                num_workers = 4,
                out_dim = 5,
                init_lr = 3e-4,
                warmup_factor = 10,
                warmup_epo = 1,
                n_epochs = 30))

device = torch.device('cuda')
# -


# # Create Folds

skf = StratifiedKFold(5, shuffle=True, random_state=42)
df_train['fold'] = -1
for i, (train_idx, valid_idx) in enumerate(skf.split(df_train, df_train['isup_grade'])):
    df_train.loc[valid_idx, 'fold'] = i


# # Model

class Model(nn.Module):
    def __init__(self, backbone, out_dim):
        super().__init__()
        self.enet = EfficientNet.from_pretrained(backbone)
        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x


# # Dataset

all_tiles = [[f'{TRAIN}{f}_{i}.png' for i in range(n_tiles)] for f in df_train.image_id]

all_files = set(str(p) for p in Path(TRAIN).iterdir())

class PANDADataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 n_tiles=n_tiles,
                 tile_mode=0,
                 rand=False,
                 transform=None,
                ):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.n_tiles = n_tiles
        self.tile_mode = tile_mode
        self.rand = rand
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id
        
        tiles = all_tiles[index]

        if self.rand:
            idxes = np.random.choice(list(range(self.n_tiles)), self.n_tiles, replace=False)
        else:
            idxes = list(range(self.n_tiles))

        n_row_tiles = int(np.sqrt(self.n_tiles))
        images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w
                
                if len(tiles) > idxes[i] and tiles[idxes[i]] in all_files:
                    this_img = open_image(tiles[idxes[i]])
                else:
                     this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                this_img = 255 - this_img
                if self.transform is not None:
                    this_img = self.transform(image=this_img)['image'] #['image']
                h1 = h * image_size
                w1 = w * image_size
                images[h1:h1+image_size, w1:w1+image_size] = this_img

        if self.transform is not None:
            images = self.transform(image=images)['image']
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)

        label = np.zeros(5).astype(np.float32)
        label[:row.isup_grade] = 1.
        return torch.tensor(images), torch.tensor(label)


# # Augmentations

transforms_train = albumentations.Compose([
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
])
transforms_val = albumentations.Compose([])

criterion = nn.BCEWithLogitsLoss()

def train_epoch(loader, optimizer, scheduler):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for i, (data, target) in enumerate(bar):        
        data, target = data.to(device), target.to(device)
        loss_func = criterion
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
        if i % 20 == 0:
            wandb.log({'smooth_loss': smooth_loss})
    return train_loss


def val_epoch(loader, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []

    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device), target.to(device)
            logits = model(data)

            loss = criterion(logits, target)

            pred = logits.sigmoid().sum(1).detach().round()
            LOGITS.append(logits)
            PREDS.append(pred)
            TARGETS.append(target.sum(1))

            val_loss.append(loss.detach().cpu().numpy())
        val_loss = np.mean(val_loss)

    LOGITS = torch.cat(LOGITS).cpu().numpy()
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    acc = (PREDS == TARGETS).mean() * 100.
    
    qwk = cohen_kappa_score(PREDS, TARGETS, weights='quadratic')
    qwk_k = cohen_kappa_score(PREDS[df_valid['data_provider'] == 'karolinska'], df_valid[df_valid['data_provider'] == 'karolinska'].isup_grade.values, weights='quadratic')
    qwk_r = cohen_kappa_score(PREDS[df_valid['data_provider'] == 'radboud'], df_valid[df_valid['data_provider'] == 'radboud'].isup_grade.values, weights='quadratic')
    print('qwk', qwk, 'qwk_k', qwk_k, 'qwk_r', qwk_r)
    wandb.log({'qwk':qwk, 'qwk_k':qwk_k, 'qwk_r':qwk_r})

    if get_output:
        return LOGITS
    else:
        return val_loss, acc, qwk

    
# -

# # Create Dataloader & Model & Optimizer

# +
train_idx = np.where((df_train['fold'] != fold))[0]
valid_idx = np.where((df_train['fold'] == fold))[0]

df_this  = df_train.loc[train_idx]
df_valid = df_train.loc[valid_idx]

dataset_train = PANDADataset(df_this , image_size, n_tiles, transform=transforms_train)
dataset_valid = PANDADataset(df_valid, image_size, n_tiles, transform=transforms_val)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=RandomSampler(dataset_train), num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, sampler=SequentialSampler(dataset_valid), num_workers=num_workers)

model = Model(enet_type, out_dim=out_dim)
model = model.to(device)
wandb.watch(model)

optimizer = optim.Adam(model.parameters(), lr=init_lr/warmup_factor)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs-warmup_epo)
scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

print(len(dataset_train), len(dataset_valid))
# -

# # Run Training

# +

qwk_max = 0.
best_file = f'{enet_type}_best_fold{fold}.pth'
for epoch in range(1, n_epochs+1):
    print(time.ctime(), 'Epoch:', epoch)

    train_loss = train_epoch(train_loader, optimizer, scheduler)
    val_loss, acc, qwk = val_epoch(valid_loader)

    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, val loss: {np.mean(val_loss):.5f}, acc: {(acc):.5f}, qwk: {(qwk):.5f}'
    print(content)
    with open(f'log_{enet_type}.txt', 'a') as appender:
        appender.write(content + '\n')

    if qwk > qwk_max:
        print('score2 ({:.6f} --> {:.6f}).  Saving model ...'.format(qwk_max, qwk))
        torch.save(model.state_dict(), best_file)
        qwk_max = qwk

torch.save(model.state_dict(), os.path.join(f'{enet_type}_final_fold{fold}.pth'))
# -

# # My Local Train Log
#
#
# ```
# Tue June 2 15:39:21 2020 Epoch 1, lr: 0.0000300, train loss: 0.42295, val loss: 0.29257, acc: 47.50471, qwk: 0.77941
# Tue June 2 15:51:56 2020 Epoch 2, lr: 0.0003000, train loss: 0.34800, val loss: 0.48723, acc: 29.09605, qwk: 0.58493
# Tue June 2 16:04:28 2020 Epoch 3, lr: 0.0003000, train loss: 0.29207, val loss: 0.27091, acc: 52.49529, qwk: 0.81714
# Tue June 2 16:17:01 2020 Epoch 4, lr: 0.0002965, train loss: 0.26521, val loss: 0.26736, acc: 57.15631, qwk: 0.80364
# Tue June 2 16:29:33 2020 Epoch 5, lr: 0.0002921, train loss: 0.24412, val loss: 0.24422, acc: 56.07345, qwk: 0.84068
# Tue June 2 16:42:05 2020 Epoch 6, lr: 0.0002861, train loss: 0.23085, val loss: 0.25306, acc: 58.05085, qwk: 0.84429
# Tue June 2 16:54:38 2020 Epoch 7, lr: 0.0002785, train loss: 0.21998, val loss: 0.21920, acc: 62.14689, qwk: 0.86278
# Tue June 2 17:07:10 2020 Epoch 8, lr: 0.0002694, train loss: 0.21062, val loss: 0.23400, acc: 61.91149, qwk: 0.86170
# Tue June 2 17:19:47 2020 Epoch 9, lr: 0.0002589, train loss: 0.20040, val loss: 0.27417, acc: 57.10923, qwk: 0.81771
# Tue June 2 17:32:25 2020 Epoch 10, lr: 0.0002471, train loss: 0.18900, val loss: 0.26732, acc: 64.92467, qwk: 0.84131
# Tue June 2 17:45:05 2020 Epoch 11, lr: 0.0002342, train loss: 0.18640, val loss: 0.21936, acc: 63.27684, qwk: 0.86580
# Tue June 2 17:57:42 2020 Epoch 12, lr: 0.0002203, train loss: 0.17387, val loss: 0.22863, acc: 61.25235, qwk: 0.86871
# Tue June 2 18:10:23 2020 Epoch 13, lr: 0.0002055, train loss: 0.16491, val loss: 0.23071, acc: 66.85499, qwk: 0.87892
# Tue June 2 18:23:00 2020 Epoch 14, lr: 0.0001901, train loss: 0.15448, val loss: 0.24338, acc: 68.45574, qwk: 0.87342
# Tue June 2 18:35:39 2020 Epoch 15, lr: 0.0001743, train loss: 0.14536, val loss: 0.22043, acc: 65.11299, qwk: 0.87169
# Tue June 2 18:48:18 2020 Epoch 16, lr: 0.0001581, train loss: 0.13918, val loss: 0.22007, acc: 67.65537, qwk: 0.88284
# Tue June 2 19:00:55 2020 Epoch 17, lr: 0.0001419, train loss: 0.13121, val loss: 0.24287, acc: 66.71375, qwk: 0.86357
# Tue June 2 19:13:35 2020 Epoch 18, lr: 0.0001257, train loss: 0.12249, val loss: 0.21583, acc: 66.80791, qwk: 0.88478
# Tue June 2 19:26:14 2020 Epoch 19, lr: 0.0001099, train loss: 0.11325, val loss: 0.21401, acc: 71.13936, qwk: 0.89178
# Tue June 2 19:38:55 2020 Epoch 20, lr: 0.0000945, train loss: 0.10602, val loss: 0.21250, acc: 70.00942, qwk: 0.89256
# Tue June 2 19:51:32 2020 Epoch 21, lr: 0.0000797, train loss: 0.09965, val loss: 0.21149, acc: 70.33898, qwk: 0.89590
# Tue June 2 20:03:59 2020 Epoch 22, lr: 0.0000658, train loss: 0.09425, val loss: 0.22203, acc: 70.76271, qwk: 0.89493
# Tue June 2 20:16:28 2020 Epoch 23, lr: 0.0000529, train loss: 0.08843, val loss: 0.22948, acc: 71.70433, qwk: 0.89304
# Tue June 2 20:28:56 2020 Epoch 24, lr: 0.0000411, train loss: 0.08448, val loss: 0.21200, acc: 71.18644, qwk: 0.89947
# Tue June 2 20:41:25 2020 Epoch 25, lr: 0.0000306, train loss: 0.07898, val loss: 0.21873, acc: 72.55179, qwk: 0.90021
# Tue June 2 20:53:53 2020 Epoch 26, lr: 0.0000215, train loss: 0.07369, val loss: 0.21842, acc: 72.64595, qwk: 0.90240
# Tue June 2 21:06:20 2020 Epoch 27, lr: 0.0000139, train loss: 0.07264, val loss: 0.21501, acc: 73.21092, qwk: 0.90450
# Tue June 2 21:18:49 2020 Epoch 28, lr: 0.0000079, train loss: 0.06950, val loss: 0.21616, acc: 73.35217, qwk: 0.90264
# Tue June 2 21:31:16 2020 Epoch 29, lr: 0.0000035, train loss: 0.06787, val loss: 0.21195, acc: 73.11676, qwk: 0.90434
# Tue June 2 21:43:43 2020 Epoch 30, lr: 0.0000009, train loss: 0.06801, val loss: 0.21014, acc: 73.11676, qwk: 0.90468
# ```

# # Thank you for reading!


