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

    # def _get_img_seg_fls(self):
    #     """
    #
    #     This private method pairs between the image files and their corresponding segmentations.
    #     The files in the folders are assumed to have last self.n_idx_chrs characters of their name
    #     an index which should match for all the (img, seg)
    #
    #     :return: list
    #     """
    #     # - All the image files
    #     img_fls = list()
    #     for root, _, image_files in os.walk(self.img_dir):
    #         for image_file in image_files:
    #             img_fls.append(f'{root}/{image_file}')
    #
    #     # - All the segmentation files
    #     seg_fls = list()
    #     for root, _, seg_files in os.walk(self.seg_dir):
    #         for seg_file in seg_files:
    #             seg_fls.append(f'{root}/{seg_file}')
    #
    #     # - Pair the image files to their segmentation files
    #     img_seg_fls = list()
    #     for img_fl, seg_fl in zip(img_fls, seg_fls):
    #
    #         img_name = img_fl[:img_fl.index('.')].split('/')[-1]
    #         img_idx = img_name[-self.n_idx_chrs:]  # => The last self.n_idx_chrs characters are the image index
    #
    #         seg_name = seg_fl[:seg_fl.index('.')].split('/')[-1]
    #         seg_idx = seg_name[-self.n_idx_chrs:]  # => The last self.n_idx_chrs characters are the image index
    #
    #         # -1- If the file indices match
    #         if img_idx == seg_idx:
    #             img_seg_fls.append((img_fl, seg_fl))
    #
    #     return img_seg_fls

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
        # seg_crps_btch = list()
        mod_seg_crps_btch = list()
        trgt_seg_msrs_btch = list()

        # 1) For each file in the batch files
        btch_fls = list(map(aux_funcs.decode_file, batch_files))
        for idx, (img_fl, seg_fl) in enumerate(btch_fls):

            # 2) Read the image and apply preprocessing functions on it
            img = preprocess_image(load_image(img_fl))

            # 2.1) No need to apply the preprocessing on the label, as it consists from a running index
            seg = load_image(seg_fl)

            aug_img_crp, aug_seg_crp, spoiled_aug_seg_crp = augment(image=img, segmentation=seg)
            trgt_seg_msr = get_seg_measure(
                ground_truth=aug_seg_crp,
                segmentation=spoiled_aug_seg_crp
            )

            img_crps_btch.append(aug_img_crp)
            # seg_crps_btch.append(aug_seg_crp)
            mod_seg_crps_btch.append(spoiled_aug_seg_crp)
            trgt_seg_msrs_btch.append(trgt_seg_msr)

        img_crps_btch = np.array(img_crps_btch, dtype=np.float32)
        mod_seg_crps_btch = np.array(mod_seg_crps_btch, dtype=np.float32)
        trgt_seg_msrs_btch = np.array(trgt_seg_msrs_btch, dtype=np.float32)

        if self.shuff:
            rnd_idxs = np.arange(img_crps_btch.shape[0])
            np.random.shuffle(rnd_idxs)

            img_crps_btch = img_crps_btch[rnd_idxs]
            mod_seg_crps_btch = mod_seg_crps_btch[rnd_idxs]
            trgt_seg_msrs_btch = trgt_seg_msrs_btch[rnd_idxs]

        return (
            tf.convert_to_tensor(img_crps_btch, dtype=tf.float32),
            tf.convert_to_tensor(mod_seg_crps_btch, dtype=tf.float32),
            tf.convert_to_tensor(trgt_seg_msrs_btch, dtype=tf.float32)
        )
