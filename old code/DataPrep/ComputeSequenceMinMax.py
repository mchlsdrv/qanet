import cv2
import numpy as np
import os
import shutil
import pickle
from scipy.ndimage.morphology import distance_transform_edt
from utils import logprint as _logprint
import matplotlib.pyplot as plt

VERBOSE = False


def logprint(string, verbose=VERBOSE):
    _logprint(string, verbose)


root_dir = '/newdisk/arbellea/Data/ISBI/Training'
# root_dir = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Challenge'
out_dir = '/HOME/CTC_Data'
# root_dir = '/Users/aarbelle/Documents/CellTrackingChallenge/Training'
# out_dir = '/Users/aarbelle/Documents/Data'
base_images_dir_format = os.path.join(root_dir, '{dataset}', '{sequence}')
base_seg_dir_format = os.path.join(root_dir, '{dataset}', '{sequence}_GT', 'SEG')
base_images_format = os.path.join(root_dir, '{dataset}', '{sequence}', 't{time:03d}.tif')
base_seg_format = os.path.join(root_dir, '{dataset}', '{sequence}_GT', 'SEG', 'man_seg{time:03d}.tif')

sets_2d = [subdir for subdir in os.listdir(root_dir) if
           (os.path.isdir(os.path.join(root_dir, subdir)) and ('2D' in subdir))]
info = {}
im = None
for data_set in sets_2d:

    for sequence in ['01', '02']:
        logprint('Dataset: {}, Sequence: {}'.format(data_set, sequence), verbose=True)
        dataset_min = 2 ** 16
        dataset_max = 0
        # out_dir_seq = os.path.join(out_dir, '{}-{}'.format(data_set, sequence))
        # out_dir_seq_raw = os.path.join(out_dir_seq, 'Raw')
        # out_dir_seq_seg = os.path.join(out_dir_seq, 'Seg')
        # out_dir_seq_dist = os.path.join(out_dir_seq, 'Dist')
        # os.makedirs(out_dir_seq_raw, exist_ok=True)
        # os.makedirs(out_dir_seq_seg, exist_ok=True)
        # os.makedirs(out_dir_seq_dist, exist_ok=True)
        all_image_files = os.listdir(base_images_dir_format.format(dataset=data_set, sequence=sequence))
        # all_seg_files = os.listdir(base_seg_dir_format.format(dataset=data_set, sequence=sequence))
        t = 0
        h = h2 = h3 = 0

        while True:
            if 't{time:03d}.tif'.format(time=t) in all_image_files:

                im_fname = base_images_format.format(dataset=data_set, sequence=sequence, time=t)
                # shutil.copy(im_fname, out_dir_seq_raw)
                # logprint('Copied frame: {}'.format(im_fname))
                logprint('read image {}'.format(t))
                im = cv2.imread(im_fname, -1)
                logprint('Done read image {}'.format(t))
                # im2 = cv2.imread(im_fname.replace('Challenge', 'Training'), -1)
                # im2_norm = (im2-29)/
                # print(im.mean(), im.std())
                # print(im2.mean(), im2.std())
                # b = plt.hist(im2.flatten(), np.arange(0, 256))
                # h2 += b[0]
                b = np.histogram(im.flatten(), np.iinfo(im.dtype).max + 1, range=[0, np.iinfo(im.dtype).max + 1])
                logprint('Done Hist image {}'.format(t))
                h += b[0].astype(np.float32) / len(all_image_files)
                logprint('Done add hist image {}'.format(t))
                dataset_min = np.minimum(dataset_min, im.min())
                dataset_max = np.maximum(dataset_max, im.max())


            else:
                break
            t += 1
        mu = np.dot(h, b[1][:-1]) / h.sum()
        sig = np.sqrt(np.dot(h, (b[1][:-1]) ** 2) / h.sum() - mu ** 2)
        info['{}-{}'.format(data_set, sequence)] = {'min': dataset_min, 'max': dataset_max, 'shape': im.shape, 'mu': mu,
                                                    'sig': sig}
        logprint(
            'Dataset: {}-{}: Min: {} Max: {}, Mu: {}, Sig: {}'.format(data_set, sequence, dataset_min, dataset_max, mu,
                                                                      sig), verbose=True)
with open(os.path.join(out_dir, 'dataset2D_info.pickle'), 'wb') as f:
    pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)
