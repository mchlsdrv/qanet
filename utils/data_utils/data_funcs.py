import os

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import tensorflow as tf
from importlib import reload
# os.chdir('D:/University/PhD/QANET/qanet')
os.chdir('C:/Users/mchls/Desktop/University/PhD/Projects/QANet/qanet')
from utils.image_utils import (
    image_funcs,
    augmentation_funcs,
    preprocessing_funcs
)
from utils.general_utils import (
    aux_funcs,
)


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, data: dict, crop_shape: tuple, batch_size: int, shuffle: bool = True):

        # FILES
        self.img_dir = data.get('images_dir')
        self.seg_dir = data.get('segmentations_dir')
        self.img_seg_fls = self._get_img_seg_fls()
        self.n_fls = len(self.img_seg_fls)

        # CROP SHAPE
        self.crp_shp = crop_shape

        # BATCH
        self.btch_sz = batch_size

        # TRAINING
        self.shuff = shuffle

    def _get_img_seg_fls(self):
        """

        This private method pairs between the image files and their corresponding segmentations.
        The files in the folders are assumed to have last three characters of their name
        an index which should match for all the triplets of (img, seg)

        :return: list
        """

        # - All the image files
        img_fls = list()
        for root, _, image_files in os.walk(self.img_dir):
            for image_file in image_files:
                img_fls.append(f'{root}/{image_file}')

        # - All the segmentation files
        seg_fls = list()
        for root, _, seg_files in os.walk(self.seg_dir):
            for seg_file in seg_files:
                seg_fls.append(f'{root}/{seg_file}')

        # - Pair the image files to their segmentation files
        img_seg_fls = list()
        for img_fl, seg_fl in zip(img_fls, seg_fls):

            img_name = img_fl[:img_fl.index('.')].split('/')[-1]
            img_idx = img_name[-3:]  # => The last 3 characters are the image index

            seg_name = seg_fl[:seg_fl.index('.')].split('/')[-1]
            seg_idx = seg_name[-3:]  # => The last 3 characters are the image index

            # -1- If the file indices match
            if img_idx == seg_idx:
                img_seg_fls.append((img_fl, seg_fl))

        return img_seg_fls

    def __len__(self):
        """
        > Returns the number of batches
        """
        return int(np.floor(self.n_fls / self.btch_sz))

    def __getitem__(self, index):
        """
        > Returns a batch of in a form of tuple:
            (image crops, segmentation crops, augmented segmentation crops, target seg measure of the crops)
        """
        if index + self.btch_sz <= self.n_fls:
            # -1- If there are enough files to fit in the batch
            btch_fls = self.img_seg_fls[index:index+self.btch_sz]
        else:
            # -2- If there are no btch_sz files left
            btch_fls = self.img_seg_fls[index:]
        return self._get_batch(batch_files=btch_fls)

    def _get_batch(self, batch_files: list):
        img_crps_btch = list()
        seg_crps_btch = list()
        mod_seg_crps_btch = list()
        trgt_seg_msrs_btch = list()

        # 1) For each file in the batch files
        btch_fls = list(map(aux_funcs.decode_file, batch_files))
        for idx, (img_fl, seg_fl) in enumerate(btch_fls):
            # 2) Read the image
            img = image_funcs.load_image(img_fl)  # cv2.imread(img_fl, cv2.IMREAD_UNCHANGED)
            # print(img.shape)
            seg = image_funcs.load_image(seg_fl)  # cv2.imread(seg_fl, cv2.IMREAD_UNCHANGED)
            # print(seg.shape)

            # 3) Apply the preprocessing function
            img = preprocessing_funcs.preprocessings(image=img)

            # 4) Randomly crop the image together with the label
            img_crp, seg_crp = image_funcs.get_random_crop(
                image=img,
                segmentation=seg,
                crop_shape=self.crp_shp
            )
            mod_seg_crp = augmentation_funcs.augmentations(seg_crp)
            trgt_seg_msr = aux_funcs.get_seg_measure(
                ground_truth=seg_crp,
                segmentations=mod_seg_crp
            )

            img_crps_btch.append(img_crp)
            seg_crps_btch.append(seg_crp)
            mod_seg_crps_btch.append(mod_seg_crp)
            trgt_seg_msrs_btch.append(trgt_seg_msr)

        img_crps_btch = np.array(img_crps_btch, dtype=np.float32)
        seg_crps_btch = np.array(seg_crps_btch, dtype=np.float32)
        mod_seg_crps_btch = np.array(mod_seg_crps_btch, dtype=np.float32)
        trgt_seg_msrs_btch = np.array(trgt_seg_msrs_btch, dtype=np.float32)

        if self.shuff:
            rnd_idxs = np.arange(img_crps_btch.shape[0])
            np.random.shuffle(rnd_idxs)

            img_crps_btch = img_crps_btch[rnd_idxs]
            seg_crps_btch = seg_crps_btch[rnd_idxs]
            mod_seg_crps_btch = mod_seg_crps_btch[rnd_idxs]
            trgt_seg_msrs_btch = trgt_seg_msrs_btch[rnd_idxs]

        return (
            tf.convert_to_tensor(img_crps_btch, dtype=tf.float32),
            tf.convert_to_tensor(seg_crps_btch, dtype=tf.float32),
            tf.convert_to_tensor(mod_seg_crps_btch, dtype=tf.float32),
            tf.convert_to_tensor(trgt_seg_msrs_btch, dtype=tf.float32)
        )


# DATA_DIR = pathlib.Path('D:/University/PhD/QANET/Data/Fluo-N2DH-GOWT1-ST')
DATA_DIR = pathlib.Path('C:/Users/mchls/Desktop/University/PhD/Projects/QANet/Data/Silver_GT/Fluo-N2DH-GOWT1-ST')
IMAGE_DIR = DATA_DIR / '01'
IMAGE_DIR.is_dir()
GT_DIR = DATA_DIR / '01_ST/SEG'
GT_DIR.is_dir()

if __name__ == '__main__':
    test_data = dict(
        images_dir=IMAGE_DIR,
        segmentations_dir=GT_DIR
    )
    reload(aux_funcs);
    reload(preprocessing_funcs);
    reload(image_funcs);
    dl = DataLoader(
        data=test_data,
        crop_shape=(254, 254, 1),
        batch_size=10,
        shuffle=True
    )
    # len(dl)
    np.count_nonzero(np.array([1, 2, 3]))
    for idx, btch in enumerate(dl):
        imgs, segs, mod_segs, J = btch
        print(J)
        plt.imshow(imgs[0], cmap='gray')
        # print(idx)
        # print(btch[0].shape)
