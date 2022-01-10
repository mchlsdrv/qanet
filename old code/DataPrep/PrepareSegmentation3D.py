import cv2
import numpy as np
import os
import shutil
import tifffile
import pickle
from scipy.ndimage.morphology import binary_erosion, distance_transform_edt
from utils import logprint as _logprint
from utils import read_multi_tiff

VERBOSE = False


def logprint(string, verbose=VERBOSE):
    _logprint(string, verbose)


def label2class(labeled_im):
    out = np.zeros_like(labeled_im)
    for label in np.unique(labeled_im):
        if label == 0:
            continue
        bw = np.equal(labeled_im, label).astype(np.float32)
        bw_erode = binary_erosion(bw, np.ones((3, 3)))
        out[np.greater(bw, 0)] = 2
        out[np.greater(bw_erode, 0)] = 1
    return out


def label2class_and_dist(labeled_im):
    out = np.zeros_like(labeled_im)
    im_shape = labeled_im.shape
    dist_1 = np.ones_like(labeled_im) * (im_shape[0] + im_shape[1]) + 2.
    dist_2 = dist_1 + 1.

    for label in np.unique(labeled_im):
        if label == 0:
            continue
        bw = np.equal(labeled_im, label).astype(np.float32)
        bw_erode = cv2.erode(bw, np.ones((3, 3)))
        out[np.greater(bw, 0)] = 2
        out[np.greater(bw_erode, 0)] = 1

        dist = distance_transform_edt(np.logical_not(bw))
        is_first_dist = np.less(dist, dist_1)
        dist_2[is_first_dist] = dist_1[is_first_dist]
        is_second_dist = np.logical_and(np.less(dist, dist_2), np.logical_not(is_first_dist))

        dist_1[is_first_dist] = dist[is_first_dist]
        dist_2[is_second_dist] = dist[is_second_dist]

    return out, (dist_1, dist_2)


def label2class3d(labled_vol):
    out = np.zeros_like(labled_vol)
    im_shape = labled_vol.shape
    dist_1 = np.ones_like(labled_vol) * (im_shape[0] + im_shape[1] + im_shape[2]) + 2.
    dist_2 = dist_1 + 1.
    for label in np.unique(labled_vol):
        if label == 0:
            continue
        bw = np.equal(labled_vol, label).astype(np.float32)
        bw_erode = binary_erosion(bw, np.ones((3, 3, 3)))
        out[np.greater(bw, 0)] = 2
        out[np.greater(bw_erode, 0)] = 1
        dist = distance_transform_edt(np.logical_not(bw))
        is_first_dist = np.less(dist, dist_1)
        dist_2[is_first_dist] = dist_1[is_first_dist]
        is_second_dist = np.logical_and(np.less(dist, dist_2), np.logical_not(is_first_dist))

        dist_1[is_first_dist] = dist[is_first_dist]
        dist_2[is_second_dist] = dist[is_second_dist]

    return out, (dist_1, dist_2)


root_dir = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training'
out_dir = '/HOME/CTC_Data'
# root_dir = '/Users/aarbelle/Documents/CellTrackingChallenge/Training'
# out_dir = '/Users/aarbelle/Documents/Data'
base_images_dir_format = os.path.join(root_dir, '{dataset}', '{sequence}')
base_seg_dir_format = os.path.join(root_dir, '{dataset}', '{sequence}_GT', 'SEG')
base_images_format = os.path.join(root_dir, '{dataset}', '{sequence}', 't{time:03d}.tif')
base_seg_format = os.path.join(root_dir, '{dataset}', '{sequence}_GT', 'SEG', 'man_seg_{time:03d}_{slice:03d}.tif')
base_seg_format_SIM = os.path.join(root_dir, '{dataset}', '{sequence}_GT', 'SEG', 'man_seg{time:03d}.tif')

sets_3d = [subdir for subdir in os.listdir(root_dir) if
           (os.path.isdir(os.path.join(root_dir, subdir)) and ('3D' in subdir))]
info = {}
im = None
for data_set in sets_3d:

    for sequence in ['01', '02']:
        dataset_min = 2 ** 16
        dataset_max = 0
        logprint('Dataset: {}, Sequence: {}'.format(data_set, sequence), verbose=True)
        out_dir_seq = os.path.join(out_dir, '{}-{}'.format(data_set, sequence))
        out_dir_seq_raw = os.path.join(out_dir_seq, 'Raw')
        out_dir_seq_seg = os.path.join(out_dir_seq, 'Seg')
        out_dir_seq_dist = os.path.join(out_dir_seq, 'Dist')
        os.makedirs(out_dir_seq_raw, exist_ok=True)
        os.makedirs(out_dir_seq_seg, exist_ok=True)
        os.makedirs(out_dir_seq_dist, exist_ok=True)
        all_image_files = os.listdir(base_images_dir_format.format(dataset=data_set, sequence=sequence))
        all_seg_files = os.listdir(base_seg_dir_format.format(dataset=data_set, sequence=sequence))
        t = 0

        while True:
            exist_labels = False
            if 't{time:03d}.tif'.format(time=t) in all_image_files:

                im_fname = base_images_format.format(dataset=data_set, sequence=sequence, time=t)
                shutil.copy(im_fname, out_dir_seq_raw)
                logprint('Copied frame: {}'.format(im_fname))
                im = read_multi_tiff(im_fname)
                dataset_min = np.minimum(dataset_min, im.min())
                dataset_max = np.maximum(dataset_max, im.max())
                seg = -1 * np.zeros_like(im)
                first_dist = -1 * np.zeros_like(im)
                second_dist = -1 * np.zeros_like(im)
                if 'SIM' not in data_set:
                    for z, page in enumerate(im):
                        if 'man_seg_{time:03d}_{slice:03d}.tif'.format(time=t, slice=z) in all_seg_files:
                            slice_labeled = cv2.imread(
                                base_seg_format.format(dataset=data_set, sequence=sequence, time=t, slice=z), -1)
                            slice_class, (first_dist_slice, second_dist_slice) = label2class_and_dist(slice_labeled)
                            first_dist[z] = first_dist_slice
                            second_dist[z] = second_dist_slice
                            seg[z] = slice_class
                            exist_labels = True
                            pass
                    if exist_labels:
                        tifffile.imsave(os.path.join(out_dir_seq_seg, 't{:03d}.tif'.format(t)), seg.astype(np.uint8))
                        logprint('Saved Seg file: {}'.format(os.path.join(out_dir_seq_seg, 't{:03d}.tif'.format(t))))
                        np.save(os.path.join(out_dir_seq_dist, 't{:03d}.npy'.format(t)),
                                np.stack([first_dist, second_dist], axis=0))
                        logprint('Saved Dist file: {}'.format(os.path.join(out_dir_seq_dist, 't{:03d}.tif'.format(t))))

                else:
                    vol_labled = read_multi_tiff(
                        base_seg_format_SIM.format(dataset=data_set, sequence=sequence, time=t))
                    seg, (first_dist, second_dist) = label2class3d(vol_labled)
                    tifffile.imsave(os.path.join(out_dir_seq_seg, 't{:03d}.tif'.format(t)), seg.astype(np.uint8))
                    logprint('Saved Seg file: {}'.format(os.path.join(out_dir_seq_seg, 't{:03d}.tif'.format(t))))
                    np.save(os.path.join(out_dir_seq_dist, 't{:03d}.npy'.format(t)),
                            np.stack([first_dist, second_dist], axis=0))
                    logprint('Saved Dist file: {}'.format(os.path.join(out_dir_seq_dist, 't{:03d}.tif'.format(t))))

            else:
                break
            t += 1

            pass
        info['{}-{}'.format(data_set, sequence)] = {'min': dataset_min, 'max': dataset_max, 'shape': im.shape}
        logprint('Dataset: {}-{}: Min: {} Max: {}'.format(data_set, sequence, dataset_min, dataset_max), verbose=True)
with open(os.path.join(out_dir, 'dataset3D_info.pickle'), 'wb') as f:
    pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)
