import os
import numpy as np
import random
import pandas as pd
from collections import Counter, defaultdict
import torch
import random
from pathlib import Path

SIZE = 512
DATA = Path("/content/clouderizer/melanoma/data")
TRAIN = DATA / "images/full"
TEST = DATA / "test"
HAIRS = DATA / "hairs"
OUT = Path("/content/clouderizer/melanoma/out")
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


def preprocess_df(train=DATA / "train.csv", test=DATA / "test.csv"):
    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test)
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

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])): #, total=len(groups_and_y_counts):
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
        
def split_full_data():
    df_folds = pd.read_csv(f'{DATA}/external_upsampled_tabular.csv').rename({'image_name': 'image_id'}, axis=1)

    df2 = pd.read_csv(f'{DATA}/folds_13062020.csv')

    df_folds = pd.merge(df_folds, df2, on=['image_id'], how='left').iloc[:, 0:8]
    df_folds.columns = ['image_id', 'sex', 'age_approx', 'anatom_site_general_challenge', 'target', 'width', 'height', 'patient_id']

    df_folds['patient_id'] = df_folds['patient_id'].fillna(df_folds['image_id'])
    df_folds['sex'] = df_folds['sex'].fillna('unknown')
    df_folds['anatom_site_general_challenge'] = df_folds['anatom_site_general_challenge'].fillna('unknown')
    df_folds['age_approx'] = df_folds['age_approx'].fillna(round(df_folds['age_approx'].mean()))
    patient_id_2_count = df_folds[['patient_id', 'image_id']].groupby('patient_id').count()['image_id'].to_dict()

    df_folds = df_folds.set_index('image_id')

    def get_stratify_group(row):
        stratify_group = row['sex']
    #     stratify_group += f'_{row["anatom_site_general_challenge"]}'
        stratify_group += f'_{row["target"]}'
        patient_id_count = patient_id_2_count[row["patient_id"]]
        if patient_id_count > 80:
            stratify_group += f'_80'
        elif patient_id_count > 60:
            stratify_group += f'_60'
        elif patient_id_count > 50:
            stratify_group += f'_50'
        elif patient_id_count > 30:
            stratify_group += f'_30'
        elif patient_id_count > 20:
            stratify_group += f'_20'
        elif patient_id_count > 10:
            stratify_group += f'_10'
        else:
            stratify_group += f'_0'
        return stratify_group

    df_folds['stratify_group'] = df_folds.apply(get_stratify_group, axis=1)
    df_folds['stratify_group'] = df_folds['stratify_group'].astype('category').cat.codes

    df_folds.loc[:, 'fold'] = 0

    skf = stratified_group_k_fold(X=df_folds.index, y=df_folds['stratify_group'], groups=df_folds['patient_id'], k=5, seed=42)

    for fold_number, (train_index, val_index) in enumerate(skf):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

    df_folds.to_csv(DATA / 'upsample.csv')
    
    return df_folds

