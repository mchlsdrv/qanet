import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging

from configs.general_configs import SEG_DIR_POSTFIX, IMAGE_PREFIX, SEG_PREFIX, VAL_BATCH_SIZE, TEST_BATCH_SIZE, NUM_WORKERS
from utils.aux_funcs import get_train_val_split

logging.getLogger('PIL').setLevel(logging.WARNING)

__author__ = 'sidorov@post.bgu.ac.il'


class ImageDS(Dataset):
    def __init__(self, data_tuples, augs):
        self.data_tuples = data_tuples
        self.augs = augs

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, index):
        img, gt_mask, aug_mask, jaccard = self.data_tuples[index]
        aug_res = self.augs(image=img.astype(np.uint8), mask=aug_mask)
        img, mask = aug_res.get('image'), aug_res.get('mask')  #np.expand_dims(aug_res.get('mask'), 0)
        img, mask = np.expand_dims(img, 0), np.expand_dims(mask, 0)

        return torch.tensor(img, dtype=torch.float), torch.tensor(mask, dtype=torch.float), torch.tensor(jaccard, dtype=torch.float)


def get_data_loaders(data: list or np.ndarray, train_batch_size: int, train_augs, val_augs, test_augs, seg_dir_postfix: str = SEG_DIR_POSTFIX, image_prefix: str = IMAGE_PREFIX, seg_prefix: str = SEG_PREFIX, val_prop: float = .2, logger: logging.Logger = None):

    train_dl = val_dl = test_dl = None
    if train_augs is not None and val_augs is not None:
        train_data, val_data = get_train_val_split(data_list=data, val_prop=val_prop, logger=logger)

        # - Create the DataLoader object
        train_dl = DataLoader(
            ImageDS(data_tuples=train_data, augs=train_augs()),
            batch_size=train_batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            shuffle=True
        )

        if isinstance(val_data, np.ndarray) and val_data.shape[0] > 0:
            val_dl = DataLoader(
                ImageDS(data_tuples=val_data, augs=val_augs()),
                batch_size=VAL_BATCH_SIZE,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                shuffle=True
            )
    elif test_augs is not None:
        test_dl = DataLoader(
            ImageDS(data_tuples=data, augs=test_augs()),
            batch_size=TEST_BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            shuffle=True
        )

    return train_dl, val_dl, test_dl
