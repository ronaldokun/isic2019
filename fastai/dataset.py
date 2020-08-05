from data import *
from utils import *
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
    
def load_datasets(fold_number):
    df_folds = pd.read_csv(f'{DATA}/upsample.csv', index_col='image_id',
                       usecols=['image_id', 'fold', 'target'], dtype={'fold': np.byte, 'target': np.byte})
    df_test = pd.read_csv(f'{DATA}/test.csv', index_col='image_name')


    ds_train = ImageDataset(
    path=TRAIN,
    image_ids=df_folds[df_folds['fold'] != fold_number].index.values,
    labels=df_folds[df_folds['fold'] != fold_number].target.values,
    transforms=get_train_transforms(),
    )

    ds_val = ImageDataset(
        path=TRAIN,
        image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
        labels=df_folds[df_folds['fold'] == fold_number].target.values,
        transforms=get_valid_transforms(),
    )

    ds_test = ImageDataset(
        path=TEST,
        image_ids=df_test.index.values,
        transforms=get_tta_transforms(),
    )
    
    return ds_train, ds_val, ds_test
