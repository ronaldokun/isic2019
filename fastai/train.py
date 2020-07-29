# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import warnings
import time
from data import *
from transforms import *
from apex import amp
from torch import nn

# import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from efficientnet_pytorch import EfficientNet
import datetime
from fastprogress import master_bar, progress_bar
from tqdm import tqdm

# get_ipython().magic("load_ext autoreload")
# get_ipython().magic("autoreload 2")
warnings.simplefilter("ignore")


# %%
batch_size = 32


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
train_df, test_df, meta_features = preprocess_df()
train_transform, test_transform = get_transforms()


# %%
train_df.rename({"tfrecord": "fold"}, axis=1, inplace=True)
train_df.head()


# %%
class MelanomaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        imfolder: Path,
        train: bool = True,
        transforms=None,
        meta_features=None,
    ):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            imfolder (str): folder with images
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            meta_features (list): list of features with meta information, such as sex and age
            
        """
        self.df = df
        self.imfolder = imfolder
        self.transforms = transforms
        self.train = train
        self.meta_features = meta_features

    def __getitem__(self, index):
        im_path = Path(f"{self.imfolder}/{self.df.iloc[index]['image_name']}.jpg")
        x = cv2.imread(str(im_path))
        meta = np.array(
            self.df.iloc[index][self.meta_features].values, dtype=np.float32
        )

        if self.transforms:
            x = self.transforms(x)

        if self.train:
            y = self.df.iloc[index]["target"]
            return (x, meta), y
        else:
            return (x, meta)

    def __len__(self):
        return len(self.df)


# %%
class Effnet(nn.Module):
    def __init__(self, arch, n_meta_features: int):
        super().__init__()
        self.arch = arch
        self.arch._fc = nn.Linear(in_features=1280, out_features=500, bias=True)
        self.meta = nn.Sequential(
            nn.Linear(n_meta_features, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(500, 250),  # FC layer output will have 250 features
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.ouput = nn.Linear(500 + 250, 1)

    def forward(self, inputs):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x, meta = inputs
        cnn_features = self.arch(x)
        meta_features = self.meta(meta)
        features = torch.cat((cnn_features, meta_features), dim=1)
        output = self.ouput(features)
        return output


# %%
test = MelanomaDataset(
    df=test_df,
    imfolder=TEST,
    train=False,
    transforms=train_transform,  # For TTA
    meta_features=meta_features,
)


# %%
import gc

epochs = 15  # Number of epochs to run
es_patience = (
    3  # Early Stopping patience - for how many epochs with no improvements to wait
)
TTA = 3  # Test Time Augmentation rounds

oof = np.zeros((len(train_df), 1))  # Out Of Fold predictions
preds = torch.zeros(
    (len(test), 1), dtype=torch.float32, device=device
)  # Predictions for test test

skf = KFold(n_splits=5, shuffle=True, random_state=47)


# %%
for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15)), 1):
    print("=" * 20, "Fold", fold, "=" * 20)

    train_idx = train_df.loc[train_df["fold"].isin(idxT)].index
    val_idx = train_df.loc[train_df["fold"].isin(idxV)].index

    model_path = f"/out/model_{fold}.pth"  # Path and filename to save model to
    best_val = 0  # Best validation score within this fold
    patience = es_patience  # Current patience counter
    arch = EfficientNet.from_pretrained("efficientnet-b1")
    model = Effnet(
        arch=arch, n_meta_features=len(meta_features)
    )  # New model for each fold
    # inference = True if Path(model_path).exists() else False

    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optim, mode="max", patience=1, verbose=True, factor=0.2
    # )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        max_lr=0.001,
        epochs=epochs,
        optimizer=optim,
        steps_per_epoch=int(len(train_df) / batch_size),
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100,
        base_momentum=0.90,
        max_momentum=0.95,
    )

    criterion = nn.BCEWithLogitsLoss()

    train = MelanomaDataset(
        df=train_df.iloc[train_idx].reset_index(drop=True),
        imfolder=TRAIN,
        train=True,
        transforms=train_transform,
        meta_features=meta_features,
    )
    val = MelanomaDataset(
        df=train_df.iloc[val_idx].reset_index(drop=True),
        imfolder=TRAIN,
        train=True,
        transforms=test_transform,
        meta_features=meta_features,
    )

    train_loader = DataLoader(
        dataset=train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        dataset=val, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        dataset=test, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # mb = master_bar(range(epochs))

    # if not inference:
    model = torch.load(model_path)  # Loading best model of this fold

    for epoch in range(epochs):
        start_time = time.time()
        correct = 0
        epoch_loss = 0
        model.train()

        for x, y in tqdm(train_loader, total=int(len(train) / 64)):
            x[0] = torch.tensor(x[0], device=device, dtype=torch.float32)
            x[1] = torch.tensor(x[1], device=device, dtype=torch.float32)
            y = torch.tensor(y, device=device, dtype=torch.float32)
            optim.zero_grad()
            z = model(x)
            loss = criterion(z, y.unsqueeze(1))
            loss.backward()
            optim.step()
            pred = torch.round(
                torch.sigmoid(z)
            )  # round off sigmoid to obtain predictions
            correct += (
                (pred.cpu() == y.cpu().unsqueeze(1)).sum().item()
            )  # tracking number of correctly predicted samples
            epoch_loss += loss.item()
            # mb.child.comment = f"{epoch_loss:.4f}"
        train_acc = correct / len(train_idx)

        model.eval()  # switch model to the evaluation mode
        val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
        with torch.no_grad():  # Do not calculate gradient since we are only predicting
            # Predicting on validation set
            for j, (x_val, y_val) in tqdm(
                enumerate(val_loader), total=int(len(val) / 32)
            ):
                x_val[0] = torch.tensor(x_val[0], device=device, dtype=torch.float32)
                x_val[1] = torch.tensor(x_val[1], device=device, dtype=torch.float32)
                y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
                z_val = model(x_val)
                val_pred = torch.sigmoid(z_val)
                val_preds[
                    j * val_loader.batch_size : j * val_loader.batch_size
                    + x_val[0].shape[0]
                ] = val_pred
            val_acc = accuracy_score(
                train_df.iloc[val_idx]["target"].values, torch.round(val_preds.cpu()),
            )
            val_roc = roc_auc_score(
                train_df.iloc[val_idx]["target"].values, val_preds.cpu()
            )

            print(
                "Epoch {:03}: | Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} | Val roc_auc: {:.3f} | Training time: {}".format(
                    epoch + 1,
                    epoch_loss,
                    train_acc,
                    val_acc,
                    val_roc,
                    str(datetime.timedelta(seconds=time.time() - start_time))[:7],
                )
            )

            scheduler.step(val_roc)

            if val_roc >= best_val:
                best_val = val_roc
                patience = es_patience  # Resetting patience since we have new best validation accuracy
                torch.save(model, model_path)  # Saving current best model
            else:
                patience -= 1
                if patience == 0:
                    print("Early stopping. Best Val roc_auc: {:.3f}".format(best_val))
                    break

    model = torch.load(model_path)  # Loading best model of this fold
    model.eval()  # switch model to the evaluation mode
    val_preds = torch.zeros((len(val_idx), 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        # Predicting on validation set once again to obtain data for OOF
        for j, (x_val, y_val) in tqdm(enumerate(val_loader), total=int(len(val) / 32)):
            x_val[0] = torch.tensor(x_val[0], device=device, dtype=torch.float32)
            x_val[1] = torch.tensor(x_val[1], device=device, dtype=torch.float32)
            y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
            z_val = model(x_val)
            val_pred = torch.sigmoid(z_val)
            val_preds[
                j * val_loader.batch_size : j * val_loader.batch_size
                + x_val[0].shape[0]
            ] = val_pred
        oof[val_idx] = val_preds.cpu().numpy()

        # Predicting on test set
        for _ in range(TTA):
            for i, x_test in tqdm(enumerate(test_loader), total=len(test) // 32):
                x_test[0] = torch.tensor(x_test[0], device=device, dtype=torch.float32)
                x_test[1] = torch.tensor(x_test[1], device=device, dtype=torch.float32)
                z_test = model(x_test)
                z_test = torch.sigmoid(z_test)
                preds[
                    i * test_loader.batch_size : i * test_loader.batch_size
                    + x_test[0].shape[0]
                ] += z_test
        preds /= TTA

    # del train, val, train_loader, val_loader, x, y, x_val, y_val
    gc.collect()

preds /= skf.n_splits


# %%
# Saving OOF predictions so stacking would be easier
pd.Series(oof.reshape(-1,)).to_csv("oof.csv", index=False)
sub = pd.read_csv(DATA / "sample_submission.csv")
sub["target"] = preds.cpu().numpy().reshape(-1,)
sub.to_csv("submission.csv", index=False)


# %%
get_ipython().system(
    'kaggle competitions submit -c siim-isic-melanoma-classification -f submission.csv -m "Melanoma Starter Image Size 384"'
)


# %%

