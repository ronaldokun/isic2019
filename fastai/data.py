import os
import pandas as pd
import numpy as np
import torch
import random
from pathlib import Path

SIZE = 384
DATA = Path("/content/clouderizer/melanoma/data")
TRAIN = DATA / "train"
TEST = DATA / "test"
HAIRS = DATA / "hairs"
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(47)


def preprocess_df():
    train_df = pd.read_csv(DATA / "train.csv")
    test_df = pd.read_csv(DATA / "test.csv")
    # One-hot encoding of anatom_site_general_challenge feature
    concat = pd.concat(
        [
            train_df["anatom_site_general_challenge"],
            test_df["anatom_site_general_challenge"],
        ],
        ignore_index=True,
    )
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix="site")
    train_df = pd.concat([train_df, dummies.iloc[: train_df.shape[0]]], axis=1)
    test_df = pd.concat(
        [test_df, dummies.iloc[train_df.shape[0] :].reset_index(drop=True)], axis=1
    )

    # Sex features
    train_df["sex"] = train_df["sex"].map({"male": 1, "female": 0})
    test_df["sex"] = test_df["sex"].map({"male": 1, "female": 0})
    train_df["sex"] = train_df["sex"].fillna(-1)
    test_df["sex"] = test_df["sex"].fillna(-1)

    # Age features
    train_df["age_approx"] /= train_df["age_approx"].max()
    test_df["age_approx"] /= test_df["age_approx"].max()
    train_df["age_approx"] = train_df["age_approx"].fillna(0)
    test_df["age_approx"] = test_df["age_approx"].fillna(0)

    train_df["patient_id"] = train_df["patient_id"].fillna(0)

    meta_features = ["sex", "age_approx"] + [
        col for col in train_df.columns if "site_" in col
    ]
    meta_features.remove("anatom_site_general_challenge")

    return train_df, test_df, meta_features
