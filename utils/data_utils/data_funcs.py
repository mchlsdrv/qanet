import numpy as np
import tensorflow as tf

from utils.image_utils.preprocessings import (
    preprocess_image,
)

from utils.image_utils.image_aux import (
    get_seg_measure,
    load_image,
)

from utils.general_utils import (
    aux_funcs,
)

from utils.image_utils.augmentations import (
    augment,
)

from configs.general_configs import (
    CROP_SIZE,
    BATCH_SIZE,
    SHUFFLE,
    NUMBER_INDEX_CHARACTERS,
)


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, data_files: list):

        # FILES
        self.n_idx_chrs = NUMBER_INDEX_CHARACTERS
        self.img_seg_fls = data_files
        self.n_fls = len(self.img_seg_fls)

        # CROP SHAPE
        self.crp_shp = (CROP_SIZE, CROP_SIZE)

        # BATCH
        self.btch_sz = BATCH_SIZE

        # TRAINING
        self.shuff = SHUFFLE

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
        aug_seg_crps_btch = list()
        trgt_seg_msrs_btch = list()

        # 1) For each file in the batch files
        btch_fls = list(map(aux_funcs.decode_file, batch_files))
        for idx, (img_fl, seg_fl) in enumerate(btch_fls):

            # 2) Read the image and apply preprocessing functions on it
            img = preprocess_image(load_image(img_fl))

            # 2.1) No need to apply the preprocessing on the label, as it consists from a running index
            seg = load_image(seg_fl)

            img_crp, seg_crp, aug_seg_crp = augment(image=img, segmentation=seg)
            trgt_seg_msr = get_seg_measure(
                ground_truth=seg_crp,
                segmentation=aug_seg_crp
            )

            img_crps_btch.append(img_crp)
            seg_crps_btch.append(seg_crp)
            aug_seg_crps_btch.append(aug_seg_crp)
            trgt_seg_msrs_btch.append(trgt_seg_msr)

        img_crps_btch = np.array(img_crps_btch, dtype=np.float32)
        seg_crps_btch = np.array(seg_crps_btch, dtype=np.float32)
        aug_seg_crps_btch = np.array(aug_seg_crps_btch, dtype=np.float32)
        trgt_seg_msrs_btch = np.array(trgt_seg_msrs_btch, dtype=np.float32)

        if self.shuff:
            rnd_idxs = np.arange(img_crps_btch.shape[0])
            np.random.shuffle(rnd_idxs)

            img_crps_btch = img_crps_btch[rnd_idxs]
            seg_crps_btch = seg_crps_btch[rnd_idxs]
            aug_seg_crps_btch = aug_seg_crps_btch[rnd_idxs]
            trgt_seg_msrs_btch = trgt_seg_msrs_btch[rnd_idxs]

        return (
            tf.convert_to_tensor(img_crps_btch, dtype=tf.float32),
            tf.convert_to_tensor(seg_crps_btch, dtype=tf.float32),
            tf.convert_to_tensor(aug_seg_crps_btch, dtype=tf.float32),
            tf.convert_to_tensor(trgt_seg_msrs_btch, dtype=tf.float32)
        )
