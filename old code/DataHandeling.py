import random

import h5py
import tensorflow as tf
import os
import glob
import cv2
import queue
import threading
import numpy as np
import pickle
import utils
import time
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation, binary_erosion, generate_binary_structure, iterate_structure
from scipy.ndimage import grey_dilation, affine_transform, grey_erosion, grey_closing, grey_opening
from scipy.ndimage.measurements import label
import time
import pandas as pd
from skimage.measure import regionprops


import warnings

warnings.filterwarnings("ignore")
__author__ = 'assafarbelle'


class CTCRAMReaderQANet2D(object):
    def __init__(self, sequence_folder_list, image_crop_size=(128, 128), batch_size=4,
                 queue_capacity=32, num_threads=3,
                 data_format='NCHW', dataset_percent=1., labeled_seg=False, debug_timing=False):
        self.coord = None
        self.sequence_data = {}
        self.sequence_folder_list = sequence_folder_list
        self.sub_seq_size = image_crop_size
        self.image_channel_depth = 1
        self.batch_size = batch_size
        self.queue_capacity = queue_capacity
        self.data_format = data_format
        self.num_threads = num_threads
        self.dataset_percent = dataset_percent
        self.labeled_seg = labeled_seg
        self.debug_timing = debug_timing
        self.q, self.q_stat = self._create_queues()
        np.random.seed(1)

    def get_batch(self):
        return self._batch_queues_()

    def _batch_queues_(self):
        with tf.name_scope('DataHandler'):

            image_batch, seg_batch, mod_seg_batch, jaccard_batch, fnames_batch = self.q.dequeue_many(self.batch_size)

            if self.data_format == 'NHWC':
                image_batch = tf.expand_dims(image_batch, 3)
                seg_batch = tf.expand_dims(seg_batch, 3)
                mod_seg_batch = tf.expand_dims(mod_seg_batch, 3)
            elif self.data_format == 'NCHW':
                image_batch = tf.expand_dims(image_batch, 1)
                seg_batch = tf.expand_dims(seg_batch, 1)
                mod_seg_batch = tf.expand_dims(mod_seg_batch, 1)
            else:
                raise ValueError('data fromat should be either "NHWC" or "NCHW"')

        return image_batch, seg_batch, mod_seg_batch, jaccard_batch, fnames_batch

    def _read_sequence_to_ram_(self):
        for sequence_folder in self.sequence_folder_list:
            train_set = True
            seq = None
            sequence_folder_orig = sequence_folder
            if isinstance(sequence_folder, tuple):
                if len(sequence_folder) == 2:
                    sequence_folder, seq = sequence_folder
                elif len(sequence_folder) == 3:
                    sequence_folder, seq, train_set = sequence_folder

            utils.log_print('Reading Sequence {}: {}'.format(sequence_folder, seq))
            with open(os.path.join(sequence_folder, 'metadata_{}.pickle'.format(seq)), 'rb') as fobj:
                metadata = pickle.load(fobj)

            filename_list = metadata['filelist']
            if self.dataset_percent < 1 and not self.dataset_percent == 0:
                length = int(len(filename_list) * self.dataset_percent)
                if self.dataset_percent > 0:
                    filename_list = filename_list[:length]
                else:
                    filename_list = filename_list[length:]
                metadata['filelist'] = filename_list
            img_size = metadata['shape']
            metadata['max_value'] = 0.
            all_images = []  # = np.zeros((len(filename_list), img_size[0], img_size[1]))
            all_seg = []  # np.zeros((len(filename_list), img_size[0], img_size[1]))
            all_full_seg = []  # np.zeros((len(filename_list)))
            keep_rate = 1
            original_size = 0
            downampled_size = 0
            for t, filename in enumerate(filename_list):
                # -1- Load the image
                img = cv2.imread(os.path.join(sequence_folder, filename[0]), -1)
                img = img.astype(np.float32)
                # -2- Normalize the image
                img = (img - img.mean()) / (img.std())

                full_seg = filename[3]
                original_size += 1

                # -3- Some random stuff here
                keep_seg = (np.random.rand() < keep_rate) and train_set

                full_seg = full_seg if keep_seg else False
                if full_seg:
                    downampled_size += 1

                # -4- Should do the same stuff as the commented
                if keep_seg:
                    seg = cv2.imread(os.path.join(sequence_folder, filename[1]), -1)
                else:
                    seg = np.ones(img.shape[:2]) * (-1)

                all_images.append(img)
                metadata['max_value'] = np.maximum(metadata['max_value'], img.max())
                all_seg.append(seg)
            if keep_rate < 1:
                print('Downsampling Training Segmentaiont with rate:{}. Original set size: {}. '
                      'Downsampled set size: {}'.format(keep_rate, original_size, downampled_size))

            self.sequence_data[sequence_folder_orig] = {'images': all_images, 'segs': all_seg}#,
                                                        # 'full_seg': all_full_seg, 'metadata': metadata}

    def _read_sequence_data(self):
        # -1- Randomly returns the data which was loaded to the RAM
        sequence_folder = random.choice(self.sequence_folder_list)
        return self.sequence_data[sequence_folder], sequence_folder

    def _load_and_enqueue(self, q, q_stat):
        while not self.coord.should_stop():
            all_t = None
            if self.debug_timing:
                # 1
                all_t = [time.time()]

            seq_data, sequence_folder = self._read_sequence_data()
            img_size = seq_data['metadata']['shape']
            if img_size[0] - self.sub_seq_size[0] > 0:
                crop_y = np.random.randint(0, img_size[0] - self.sub_seq_size[0])
            else:
                crop_y = 0
            if img_size[1] - self.sub_seq_size[1] > 0:
                crop_x = np.random.randint(0, img_size[1] - self.sub_seq_size[1])
            else:
                crop_x = 0

            if self.debug_timing: # 2
                all_t.append(time.time())
            indices = self._get_indices4elastic_transform(self.sub_seq_size, self.sub_seq_size[1]/64,
                                                          self.sub_seq_size[1] * 0.15)
            if self.debug_timing: # 3
                all_t.append(time.time())

            filename_idx = list(range(len(seq_data['metadata']['filelist'])))
            selected_frame = random.choice(filename_idx)
            fname = seq_data['metadata']['filelist'][selected_frame][0]
            crop_y_stop = crop_y + self.sub_seq_size[0]
            crop_x_stop = crop_x + self.sub_seq_size[1]
            img_crop = seq_data['images'][selected_frame][crop_y:crop_y_stop, crop_x:crop_x_stop].copy()
            img_max = seq_data['metadata']['max_value']
            seg_crop = seq_data['segs'][selected_frame][crop_y:crop_y_stop, crop_x:crop_x_stop].copy()
            if self.debug_timing: # 4
                all_t.append(time.time())

            if not np.any(seg_crop):
                continue

            # contrast factor between [0.5, 1.5]
            random_constrast_factor = np.random.rand() + 0.5
            # random brightness delta plus/minus 10% of maximum value
            random_brightness_delta = (np.random.rand() - 0.5) * 0.2 * img_max
            img_crop = self._adjust_contrast_(img_crop, random_constrast_factor)
            img_crop = self._adjust_brightness_(img_crop, random_brightness_delta)
            if self.debug_timing: # 5
                all_t.append(time.time())
            modified_seg_crop = seg_crop.copy()
            jaccard = 0.
            if self.debug_timing: # 6
                all_t.append(time.time())
            if not np.equal(seg_crop, -1).all():
                seg_not_valid = np.equal(seg_crop, -1)
                labeled_gt = modified_seg_crop
                labeled_gt[:, 0] = 0
                labeled_gt[:, -1] = 0
                labeled_gt[-1, :] = 0
                labeled_gt[0, :] = 0
                if self.debug_timing: # 7
                    all_t.append(time.time())

                # - ERODE -
                erode_dilate = np.random.randint(-5, 5, 1)
                if not erode_dilate == 0:
                    labeled_gt = self.erode_dilate_merge(labeled_gt, erode_dilate)
                if self.debug_timing: # 8
                    all_t.append(time.time())

                # - MERGE PROB -
                merge_prob = 0.9
                labeled_gt = self.merge_lables(labeled_gt, merge_prob)
                if self.debug_timing: # 9
                    all_t.append(time.time())

                # - ELASTIC -
                do_elastic = np.random.rand()
                if do_elastic > 0.1:
                    trans_seg = self._get_transformed_image_(labeled_gt.astype(np.float32),
                                                             indices, seg=True)
                else:
                    trans_seg = labeled_gt.copy()
                if self.debug_timing: # 10
                    all_t.append(time.time())
                jaccard = self.calc_jaccard(seg_crop, trans_seg)
                if self.debug_timing: # 11
                    all_t.append(time.time())
                if not self.labeled_seg:
                    trans_seg_fix = self.labeled2trinary(trans_seg)
                else:
                    trans_seg_fix = trans_seg
                # trans_not_valid = np.logical_or(np.greater(trans_not_valid, 0.5), np.equal(trans_seg, -1))
                if self.debug_timing: # 12
                    all_t.append(time.time())
                modified_seg_crop = trans_seg_fix
                # modified_seg_crop[trans_not_valid] = -1
                seg_crop = self.labeled2trinary(seg_crop)
                if self.debug_timing: # 13
                    all_t.append(time.time())

            flip = np.random.randint(0, 2, 2)
            if flip[0]:
                img_crop = cv2.flip(img_crop, 0)
                seg_crop = cv2.flip(seg_crop, 0)
                modified_seg_crop = cv2.flip(modified_seg_crop, 0)
            if flip[1]:
                img_crop = cv2.flip(img_crop, 1)
                seg_crop = cv2.flip(seg_crop, 1)
                modified_seg_crop = cv2.flip(modified_seg_crop, 1)

            # - ROTATE -
            rotate = np.random.randint(0, 4)
            if rotate > 0:
                img_crop = np.rot90(img_crop, rotate)
                seg_crop = np.rot90(seg_crop, rotate)
                modified_seg_crop = np.rot90(modified_seg_crop, rotate)
            if self.debug_timing: # 14
                all_t.append(time.time())
            if self.coord.should_stop():
                return

            while q_stat().numpy() > 0.9:
                if self.coord.should_stop():
                    return
                time.sleep(1)

            q.enqueue([img_crop, seg_crop, modified_seg_crop, jaccard, fname])
            if self.debug_timing: # 15
                all_t.append(time.time())
                all_t = np.array(all_t)
                print(all_t[-1]-all_t[0], all_t[1:]-all_t[:-1])




    def start_queues(self, coord=tf.train.Coordinator(), debug=False):
        self._read_sequence_to_ram_()
        threads = []
        self.coord = coord

        if debug:
            self._load_and_enqueue(self.q, self.q_stat)
        for _ in range(self.num_threads):
            t = threading.Thread(target=self._load_and_enqueue, args=(self.q, self.q_stat))
            t.daemon = True
            t.start()
            threads.append(t)
            self.coord.register_thread(t)

        t = threading.Thread(target=self._monitor_queues_)
        t.daemon = True
        t.start()
        self.coord.register_thread(t)
        threads.append(t)
        return threads

    def _create_queues(self):
        def normed_size(_q):
            @tf.function
            def normed_q_stat():
                return tf.cast(_q.size(), tf.float32) / self.queue_capacity

            return normed_q_stat

        with tf.name_scope('DataHandler'):
            dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.string]
            shapes = [self.sub_seq_size, self.sub_seq_size, self.sub_seq_size, (), ()]

            q = tf.queue.FIFOQueue(self.queue_capacity, dtypes=dtypes, shapes=shapes, name='data_q')
            q_stat = normed_size(q)

        return q, q_stat

    @staticmethod
    def _get_elastic_affine_matrix_(shape_size, alpha_affine):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
         .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
              Convolutional Neural Networks applied to Visual Document Analysis", in
              Proc. of the International Conference on Document Analysis and
              Recognition, 2003.

          Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
         """
        random_state = np.random.RandomState(None)

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        affine_matrix = cv2.getAffineTransform(pts1, pts2)

        return affine_matrix, random_state

    @staticmethod
    def _get_transformed_image_(image, indices, seg=False):

        shape = image.shape
        if seg:
            # trans_img = cv2.warpAffine(image, affine_matrix, shape[::-1], borderMode=cv2.BORDER_CONSTANT,
            #                            borderValue=-1, flags=cv2.INTER_NEAREST)
            trans_coord = map_coordinates(image, indices, order=0, mode='reflect', cval=0).reshape(shape)
            # trans_coord = map_coordinates(image, indices, order=0, mode='constant', cval=0).reshape(shape)
        else:
            # trans_img = cv2.warpAffine(image, affine_matrix, shape[::-1], borderMode=cv2.BORDER_REFLECT_101)
            trans_coord = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

        return trans_coord

    @staticmethod
    def _get_indices4elastic_transform(shape, alpha, sigma):
        dxr = np.random.rand(*shape)
        dx = gaussian_filter((dxr * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        return indices

    @staticmethod
    def labeled2trinary(trans_seg):
        # separate close blobs with the edge of transformation
        # edge is pixels that get fractional value after transofrmation
        strel = np.zeros((3, 3))
        trans_seg = np.round(trans_seg)
        # region_props = skimage.measure.regionprops(trans_seg)
        # errosion = grey_erosion(trans_seg, np.zeros(3, 3, 3))
        dilation = grey_dilation(trans_seg.astype(np.int32), structure=strel.astype(np.int8))
        bw = np.minimum(trans_seg, 1)
        bw[np.logical_and(np.not_equal(trans_seg, dilation), np.greater(dilation, 0))] = 2

        return bw

    @staticmethod
    def _gt2dist_(gt_image):
        gt_fg = gt_image == 1
        _, labeled_gt = cv2.connectedComponents(gt_fg.astype(np.uint8))
        im_shape = gt_image.shape
        dist_1 = np.ones_like(gt_image) * (im_shape[0] + im_shape[1]) + 2.
        dist_2 = dist_1 + 1.

        for this_label in np.unique(labeled_gt):
            if this_label == 0:
                continue
            bw = np.equal(labeled_gt, this_label).astype(np.float32)
            bw_erode = cv2.erode(bw, np.ones((3, 3)))
            edge = np.logical_and(np.logical_not(bw_erode), bw)

            dist = distance_transform_edt(np.logical_not(edge))
            is_first_dist = np.less(dist, dist_1)
            dist_2[is_first_dist] = dist_1[is_first_dist]
            is_second_dist = np.logical_and(np.less(dist, dist_2), np.logical_not(is_first_dist))

            dist_1[is_first_dist] = dist[is_first_dist]
            dist_2[is_second_dist] = dist[is_second_dist]
        out = np.stack((dist_1, dist_2), 0)

        return out, (dist_1, dist_2)

    @staticmethod
    def _adjust_brightness_(image, delta):
        """
        Args:
        image (numpy.ndarray)
        delta
        """

        out_img = image + delta
        return out_img

    @staticmethod
    def _adjust_contrast_(image, factor):
        """
        Args:
        image (numpy.ndarray)
        factor
        """

        img_mean = image.mean()
        out_img = (image - img_mean) * factor + img_mean
        return out_img

    @staticmethod
    def erode_dilate_merge(labled_seg, kernel_size):
        ed = np.random.rand(1) < 0.5
        strel = generate_binary_structure(2, 1)
        strel = iterate_structure(strel, kernel_size[0])

        if kernel_size > 0:
            if ed:
                labled_seg_out = grey_dilation(labled_seg, footprint=strel)
            else:
                labled_seg_out = grey_opening(labled_seg, footprint=strel)
        else:
            if ed:
                labled_seg_out = grey_erosion(labled_seg, footprint=strel)
            else:
                labled_seg_out = grey_closing(labled_seg, footprint=strel)

        return labled_seg_out

    @staticmethod
    def merge_lables(labled_seg, merge_prob=0.1):
        strel = generate_binary_structure(2, 1)
        labled_seg_dilate = grey_dilation(labled_seg, footprint=strel)
        diff = abs(labled_seg_dilate - labled_seg) > 0
        diff = np.logical_and(diff, labled_seg > 0)
        diff = np.logical_and(diff, labled_seg_dilate > 0)
        orig_labels = labled_seg[diff]
        dilated_labels = labled_seg_dilate[diff]
        pairs = set(zip(orig_labels.ravel(), dilated_labels.ravel()))
        out_labels = labled_seg.copy()
        for l1, l2 in pairs:
            if np.random.rand() < merge_prob:
                l1_out = out_labels[labled_seg == l1].min()
                l2_out = out_labels[labled_seg == l2].min()
                min_l = np.minimum(l1_out, l2_out)
                max_l = np.maximum(l1_out, l2_out)

                out_labels[out_labels == max_l] = min_l
        return out_labels

    @staticmethod
    def one_hot(labled):
        labled = labled.astype(np.uint16)
        orig_shape = labled.shape
        flat_lables = labled.ravel()  # => flattens the lables
        unique_vals, flat_lables = np.unique(flat_lables, return_inverse=True)  # => return_inverse return the indices of the values in the unique array where they should be placed to construct the original array

        # (!) This condition will throw a  "IndexError: index 0 is out of bounds for axis 0 with size 0" in case len(unique_vals) == 0
        if len(unique_vals) == 0 and unique_vals[0] > 0:
            flat_lables[:] = 1  # => Places 1 in all the entries

        out_flat = np.zeros((len(flat_lables), flat_lables.max() + 1), dtype=np.bool)
        out_flat[np.arange(len(flat_lables)), flat_lables] = True
        out_flat = out_flat[:, 1:] # (?)
        out = np.reshape(out_flat, orig_shape + (flat_lables.max(),))  # (?)
        return out, out_flat

    def calc_jaccard(self, gt, seg):
        # - Extracts the foreground, i.e., the segmentations
        no_bg = np.greater(np.maximum(gt,seg),0)  # no_bg = np.maximum(gt, seg) > 0

        # - Ground truth label
        one_hot_gt, _ = self.one_hot(gt[no_bg])
        one_hot_gt = np.expand_dims(one_hot_gt, 2)

        # - Segmentation label
        one_hot_seg, _ = self.one_hot(seg[no_bg])
        one_hot_seg = np.expand_dims(one_hot_seg, 1)

        # - For each ground truth image calculates the area
        area = np.sum(one_hot_gt, axis=(0), keepdims=True)

        # - For each segmentation image calculates the area
        area_seg = np.sum(one_hot_seg, axis=(0), keepdims=True)

        intersection = np.sum(np.logical_and(one_hot_gt, one_hot_seg), axis=(0), keepdims=True)
        union = area + area_seg - intersection

        # - Detection is considered only if the intersection is greated then 0.5
        true_detections = np.greater(np.divide(intersection, area + 0.0000001), 0.5)

        # - Calculates the number of non-zero area objects
        num_objects = np.count_nonzero(np.greater(area, 0))

        # - Calculates the jaccard
        jaccard = np.divide(intersection, union + 0.00000001)

        # - If there are any detections - calculates teh seg measure, i.e., how many of the
        # non-zero area objects overlap with more than 0.5 with the segmentation
        if np.any(true_detections):
            true_jaccard = jaccard[true_detections]
            seg_measure = np.sum(true_jaccard) / num_objects
        else:
            seg_measure = 0

        return seg_measure

    @classmethod
    def unit_test(cls):
        import matplotlib.pyplot as plt
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        root_dir = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/'

        sequence_folder_list = [(os.path.join(root_dir, 'Fluo-N2DH-SIM+'), '01'),
                                (os.path.join(root_dir, 'Fluo-N2DH-SIM+'), '02')]
        image_crop_size = (256, 256)
        batch_size = 10
        queue_capacity = 250
        num_threads = 2
        data_format = 'NCHW'
        dataset_percent = 0.7 - 1
        labeled_seg = True
        debug_timing = True
        data = cls(sequence_folder_list, image_crop_size, batch_size, queue_capacity, num_threads, data_format,
                   dataset_percent, labeled_seg, debug_timing)

        debug = True
        data.start_queues(debug=debug)
        all_jaccard = []
        for i in range(100):
            image_batch, seg_batch, modified_seg_batch, jaccard, fnames = data.get_batch()
            utils.log_print(i, image_batch.shape, seg_batch.shape, jaccard)
            all_jaccard.append(jaccard)
            pass
        all_jac = np.concatenate(all_jaccard)
        print(all_jac.mean())
        plt.figure(1)
        plt.cla()
        plt.hist(all_jac, 20, (0, 1))
        plt.show()

