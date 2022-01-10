import glob

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle
import utils

# root_dir = '/newdisk/CTC_Data'
# root_dir = '/Users/aarbelle/Documents/Data'
# root_dir, challenge_set = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/', False
root_dir, challenge_set = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Test', True
# competitor = 'CVUT-CZ'
# competitor = '_'+competitor
# with open(os.path.join(root_dir, 'dataset2D{}_info.pickle'.format(competitor)), 'rb') as f:
#     dataset_info = pickle.load(f)

sets_3d = [subdir for subdir in os.listdir(root_dir) if
           (os.path.isdir(os.path.join(root_dir, subdir)) and ('3D' in subdir and 'TRIC' not in subdir))]

for dataset in sets_3d:
    for seq in ['01', '02']:
        print('{}:{}'.format(dataset, seq))
        if os.path.isfile(os.path.join(root_dir, dataset, 'metadata_{}.pickle'.format(seq))):
            continue
            with open(os.path.join(root_dir, dataset, 'metadata_{}.pickle'.format(seq)), 'rb') as fobj:
                seq_metadata = pickle.load(fobj)
                seq_metadata_new = {'shape': seq_metadata['shape'], 'filelist': []}
                for f in seq_metadata['filelist']:
                    f0 = f[0].replace(os.path.join(root_dir, dataset), '.')
                    if isinstance(f[1], str):
                        f1 = f[1].replace(os.path.join(root_dir, dataset), '.')
                    elif isinstance(f[1], list):
                        f1 = []
                        for s_i, s in enumerate(f[1]):
                            f1.append(s.replace(os.path.join(root_dir, dataset), '.'))
                    else:
                        f1 = f[1]
                    seq_metadata_new['filelist'].append([f0, f1, f[2]])
            with open(os.path.join(root_dir, dataset, 'metadata_{}.pickle'.format(seq)), 'wb') as fobj:
                pickle.dump(seq_metadata_new, fobj, pickle.HIGHEST_PROTOCOL)
                print('Corrected: {}'.format(os.path.join(root_dir, dataset, 'metadata_{}.pickle'.format(seq))))
            continue


        seq_metadata = {}
        seq_metadata['filelist'] = []
        seq_metadata['shape'] = None

        raw_file_dir = os.path.join(root_dir, dataset, seq)
        seg_file_dir = os.path.join(root_dir, dataset, seq + '_GT', 'SEG')
        track_file_dir = os.path.join(root_dir, dataset, seq + '_GT', 'TRA')
        raw_file_template = os.path.join(raw_file_dir, 't{:03d}.tif')
        seg_file_template = os.path.join(seg_file_dir, 'man_seg_{:03d}_{:03d}.tif')
        seg_sim_file_template = os.path.join(seg_file_dir, 'man_seg{:03d}.tif')
        track_file_template = os.path.join(track_file_dir, 'man_track{:03d}.tif')
        all_image_files = glob.glob(os.path.join(raw_file_dir, '*.tif'))
        all_seg_files = glob.glob(os.path.join(seg_file_dir, '*.tif'))
        all_track_files = glob.glob(os.path.join(track_file_dir, '*.tif'))

        # will be a list of tuples holding (raw_fname, seg_fname, is_seg_val, dist_fname)

        t = 0

        while True:
            im_fname = raw_file_template.format(t)
            local_im_fname = im_fname.replace(os.path.join(root_dir, dataset), '.')

            track_fname = track_file_template.format(t)
            if im_fname in all_image_files:
                if seq_metadata['shape'] is None:
                    im = utils.read_multi_tiff(im_fname)
                    seq_metadata['shape'] = im.shape
                seg_list = None
                valid_seg = '_'
                valid_segs = []
                if challenge_set:
                    row = [local_im_fname, None, None]
                    seq_metadata['filelist'].append(row)
                    t += 1
                    continue

                if 'SIM' in dataset:
                    seg_fname = seg_sim_file_template.format(t)
                    local_seg_fname = seg_fname.replace(os.path.join(root_dir, dataset), '.')
                    row = [local_im_fname, local_seg_fname, True]
                    seq_metadata['filelist'].append(row)
                    t += 1
                    continue

                for z in range(seq_metadata['shape'][0]):

                    seg_fname = seg_file_template.format(t, z)

                    if seg_fname in all_seg_files:
                        if 'SIM' in dataset:
                            valid_seg = True
                            valid_segs.append(valid_seg)
                        else:
                            if seg_list is None:
                                seg_list = []
                            seg = cv2.imread(seg_fname, -1)
                            track = utils.read_multi_tiff(track_fname)
                            seg = np.minimum(seg, 1)
                            track = np.minimum(track, 1)
                            track_z = track[z]
                            overlap = np.sum(seg[track_z>0]) / np.sum(track_z)
                            if np.sum(track_z) == 0:
                                im = utils.read_multi_tiff(im_fname)[z]
                                im = (im - im.min()) / (im.max() - im.min())
                                imR = im.copy()
                                imG = im.copy()
                                imB = im.copy()
                                imR[np.equal(seg, 1)] = 1
                                imG[np.equal(seg, 1)] = 0
                                imB[np.equal(seg, 1)] = 0
                                imrgb = np.stack([imR, imG, imB], 2)
                                plt.figure(1)
                                plt.cla()
                                plt.imshow(imrgb)
                                plt.title('T = {}, Z= {}'.format(t, z))
                                plt.pause(0.1)
                                valid_seg = input('Is seg for frmae {}, {} valid? '.format(t, z)) == ''
                            else:
                                if overlap == 1:
                                    valid_seg = True
                                else:
                                    valid_seg = False
                            # im = (im - im.min()) / (im.max() - im.min())
                            # seg = cv2.imread(seg_fname, -1)
                            # imR = im.copy()
                            # imG = im.copy()
                            # imB = im.copy()
                            # imR[np.equal(seg, 2)] = 1
                            # imG[np.equal(seg, 2)] = 0
                            # imB[np.equal(seg, 2)] = 0
                            # imrgb = np.stack([imR, imG, imB], 2)
                            # plt.figure(1)
                            # plt.cla()
                            # plt.imshow(imrgb)
                            # plt.title('T = {}'.format(t))
                            # plt.pause(0.1)
                            # valid_seg = input('Is seg for frmae {} valid? '.format(t))
                        valid_segs.append(valid_seg)
                        local_seg_fname = seg_fname.replace(os.path.join(root_dir, dataset), '.')
                        seg_list.append(local_seg_fname)

                row = [local_im_fname, seg_list, valid_segs]

                seq_metadata['filelist'].append(row)
            else:
                break
            t += 1
        with open(os.path.join(root_dir, dataset, 'metadata_{}.pickle'.format(seq)), 'wb') as f:
            pickle.dump(seq_metadata, f, pickle.HIGHEST_PROTOCOL)
            print('Created: {}'.format(os.path.join(root_dir, dataset, 'metadata_{}.pickle'.format(seq))))
