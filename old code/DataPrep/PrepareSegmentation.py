import cv2
import numpy as np
import os
import shutil
import pickle
from scipy.ndimage.morphology import distance_transform_edt
from utils import logprint as _logprint

VERBOSE = True
competitor = 'CVUT-CZ'
competitor = '_'+competitor


def logprint(string, verbose=VERBOSE):
    _logprint(string, verbose)


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
        edge = np.logical_and(np.logical_not(bw_erode),bw)
        out[np.greater(bw, 0)] = 2
        out[np.greater(bw_erode, 0)] = 1

        # dist = distance_transform_edt(np.logical_not(edge))
        # is_first_dist = np.less(dist, dist_1)
        # dist_2[is_first_dist] = dist_1[is_first_dist]
        # is_second_dist = np.logical_and(np.less(dist, dist_2), np.logical_not(is_first_dist))
        #
        # dist_1[is_first_dist] = dist[is_first_dist]
        # dist_2[is_second_dist] = dist[is_second_dist]

    return out, (dist_1, dist_2)


root_dir = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training'
root_dir = '/newdisk/arbellea/Data/ISBI/Challenge'
out_dir = '/newdisk/CTC_Data'
# root_dir = '/Users/aarbelle/Documents/CellTrackingChallenge/Training'
# out_dir = '/Users/aarbelle/Documents/Data'
base_images_dir_format = os.path.join(root_dir, '{dataset}', '{sequence}')
base_seg_dir_format = os.path.join(root_dir, '{dataset}', '{sequence}_RES{competitor}')
base_images_format = os.path.join(root_dir, '{dataset}', '{sequence}', 't{time:03d}.tif')
base_seg_format = os.path.join(root_dir, '{dataset}', '{sequence}_GT', 'SEG', 'man_seg{time:03d}.tif')
base_seg_format = os.path.join(root_dir, '{dataset}', '{sequence}_RES{competitor}','mask{time:03d}.tif')

sets_2d = [subdir for subdir in os.listdir(root_dir) if
           (os.path.isdir(os.path.join(root_dir, subdir)) and ('2D' in subdir and 'L-HeLa' in subdir and 'copy' not in subdir))]
info = {}
im = None
for data_set in sets_2d:

    for sequence in ['01', '02']:
        logprint('Dataset: {}, Sequence: {}'.format(data_set, sequence), verbose=True)
        dataset_min = 2 ** 16
        dataset_max = 0
        out_dir_seq = os.path.join(out_dir, '{}{}-{}'.format(data_set,competitor, sequence))
        out_dir_seq_raw = os.path.join(out_dir_seq, 'Raw')
        out_dir_seq_seg = os.path.join(out_dir_seq, 'Seg')
        out_dir_seq_dist = os.path.join(out_dir_seq, 'Dist')
        os.makedirs(out_dir_seq_raw, exist_ok=True)
        os.makedirs(out_dir_seq_seg, exist_ok=True)
        os.makedirs(out_dir_seq_dist, exist_ok=True)
        all_image_files = os.listdir(base_images_dir_format.format(dataset=data_set, sequence=sequence))
        all_seg_files = os.listdir(base_seg_dir_format.format(dataset=data_set, sequence=sequence, competitor=competitor))
        t = 0

        while True:
            if 't{time:03d}.tif'.format(time=t) in all_image_files:

                im_fname = base_images_format.format(dataset=data_set, sequence=sequence, time=t)
                shutil.copy(im_fname, out_dir_seq_raw)
                logprint('Copied frame: {}'.format(im_fname))
                im = cv2.imread(im_fname, -1)

                dataset_min = np.minimum(dataset_min, im.min())
                dataset_max = np.maximum(dataset_max, im.max())


                seg = -1 * np.zeros_like(im)

                # if 'man_seg{time:03d}.tif'.format(time=t) in all_seg_files:
                if 'mask{time:03d}.tif'.format(time=t) in all_seg_files:
                    seg_labled = cv2.imread(
                        base_seg_format.format(dataset=data_set, sequence=sequence, time=t, competitor=competitor), -1)
                    seg, (first_dist, second_dist) = label2class_and_dist(seg_labled)
                    cw = np.array([np.count_nonzero(seg==s) for s in np.unique(seg)]).astype(np.float32)
                    cw = cw/cw.sum()
                    cv2.imwrite(os.path.join(out_dir_seq_seg, 't{:03d}.tif'.format(t)), seg.astype(np.uint8))
                    logprint('Saved Seg file: {}'.format(os.path.join(out_dir_seq_seg, 't{:03d}.tif'.format(t))))
                    dist_concat = np.stack([first_dist, second_dist], axis=0)
                    np.save(os.path.join(out_dir_seq_dist, 't{:03d}.npy'.format(t)), dist_concat)
                    logprint('Saved Dist file: {}'.format(os.path.join(out_dir_seq_dist, 't{:03d}.npy'.format(t))))
                    np.save(os.path.join(out_dir_seq_dist, 't_cw{:03d}.npy'.format(t)), cw)
                    logprint('Saved Class Weights file: {}'.format(os.path.join(out_dir_seq_dist, 't_cw{:03d}.npy'.format(t))))
            else:
                break
            t += 1
        info['{}{}-{}'.format(data_set,competitor, sequence)] = {'min': dataset_min, 'max': dataset_max, 'shape': im.shape}
        logprint('Dataset: {}-{}: Min: {} Max: {}'.format(data_set, sequence, dataset_min, dataset_max), verbose=True)
with open(os.path.join(out_dir, 'dataset2D{}_info.pickle'.format(competitor)), 'wb') as f:
    pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)
