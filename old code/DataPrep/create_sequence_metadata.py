
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle
import utils

root_dir = '/newdisk/CTC_Data'
# root_dir = '/Users/aarbelle/Documents/Data'
competitor = 'CVUT-CZ'
competitor = '_'+competitor
with open(os.path.join(root_dir, 'dataset2D{}_info.pickle'.format(competitor)), 'rb') as f:
    dataset_info = pickle.load(f)

sets_2d = [subdir for subdir in os.listdir(root_dir) if
           (os.path.isdir(os.path.join(root_dir, subdir)) and ('2D' in subdir and competitor in subdir and 'L-HeLa' in subdir))]

for dataset in sets_2d:
    # if os.path.exists(os.path.join(root_dir, dataset, 'metadata.pickle')):
    #     with open(os.path.join(root_dir, dataset, 'metadata.pickle'), 'rb') as f:
    #         seq_metadata = pickle.load(f)
    #         for k, v in dataset_info[dataset].items():
    #             seq_metadata[k] = v
        # with open(os.path.join(root_dir, dataset, 'metadata.pickle'), 'wb') as f:
        #     pickle.dump(seq_metadata, f, pickle.HIGHEST_PROTOCOL)
        # continue

    seq_metadata = dataset_info[dataset]
    seq_metadata['filelist'] = []  # will be a list of tuples holding (raw_fname, seg_fname, is_seg_val, dist_fname)
    raw_file_template = os.path.join(root_dir, dataset, 'Raw', 't{:03d}.tif')
    seg_file_template = os.path.join(root_dir, dataset, 'Seg', 't{:03d}.tif')
    dist_file_template = os.path.join(root_dir, dataset, 'Dist', 't{:03d}.npy')
    t = 0
    all_image_files = os.listdir(os.path.join(root_dir, dataset, 'Raw'))
    all_seg_files = os.listdir(os.path.join(root_dir, dataset, 'Seg'))
    while True:
        if 't{:03d}.tif'.format(t) in all_image_files:

            im_fname = raw_file_template.format(t)
            seg_fname = seg_file_template.format(t)
            dist_fname = dist_file_template.format(t)
            if 't{:03d}.tif'.format(t) in all_seg_files:
                if 'SIM' in dataset or competitor in dataset:
                    valid_seg = ''
                else:
                    im = cv2.imread(im_fname, -1)
                    im = (im - im.min()) / (im.max() - im.min())
                    seg = cv2.imread(seg_fname, -1)
                    imR = im.copy()
                    imG = im.copy()
                    imB = im.copy()
                    imR[np.equal(seg, 2)] = 1
                    imG[np.equal(seg, 2)] = 0
                    imB[np.equal(seg, 2)] = 0
                    imrgb = np.stack([imR, imG, imB], 2)
                    plt.figure(1)
                    plt.cla()
                    plt.imshow(imrgb)
                    plt.title('T = {}'.format(t))
                    plt.pause(0.1)
                    valid_seg = input('Is seg for frmae {} valid? '.format(t))

                row = (im_fname, seg_fname, valid_seg == '', dist_fname)
            else:
                row = (im_fname, None, None, None)
            seq_metadata['filelist'].append(row)

        else:
            break
        t += 1
    with open(os.path.join(root_dir, dataset, 'metadata.pickle'), 'wb') as f:
        pickle.dump(seq_metadata, f, pickle.HIGHEST_PROTOCOL)

sets_3d = [subdir for subdir in os.listdir(root_dir) if
           (os.path.isdir(os.path.join(root_dir, subdir)) and ('3D' in subdir))]

with open(os.path.join(root_dir, 'dataset3D_info.pickle'), 'rb') as f:
    dataset_info = pickle.load(f)

for dataset in sets_3d:
    if os.path.exists(os.path.join(root_dir, dataset, 'metadata.pickle')):
        continue
    seq_metadata = dataset_info[dataset]
    seq_metadata['filelist'] = []  # will be a list of tuples holding (raw_fname, seg_fname, is_seg_val, dist_fname)
    raw_file_template = os.path.join(root_dir, dataset, 'Raw', 't{:03d}.tif')
    seg_file_template = os.path.join(root_dir, dataset, 'Seg', 't{:03d}.tif')
    dist_file_template = os.path.join(root_dir, dataset, 'Dist', 't{:03d}.npy')
    t = 0
    all_image_files = os.listdir(os.path.join(root_dir, dataset, 'Raw'))
    all_seg_files = os.listdir(os.path.join(root_dir, dataset, 'Seg'))
    while True:
        if 't{:03d}.tif'.format(t) in all_image_files:

            im_fname = raw_file_template.format(t)
            seg_fname = seg_file_template.format(t)
            dist_fname = dist_file_template.format(t)
            if 't{:03d}.tif'.format(t) in all_seg_files:
                if 'SIM' in dataset:
                    valid_vec = True
                else:

                    im = utils.read_multi_tiff(im_fname)
                    im = (im - im.min()) / (im.max() - im.min())
                    seg = utils.read_multi_tiff(seg_fname)
                    valid_vec = np.ones(im.shape[0])
                    for p, (im_page, seg_page) in enumerate(zip(im, seg)):
                        if not np.any(np.equal(seg_page, 2)):
                            continue
                        imR = im_page.copy()
                        imG = im_page.copy()
                        imB = im_page.copy()
                        imR[np.equal(seg_page, 2)] = 1
                        imG[np.equal(seg_page, 2)] = 0
                        imB[np.equal(seg_page, 2)] = 0
                        imrgb = np.stack([imR, imG, imB], 2)
                        plt.figure(1)
                        plt.cla()
                        plt.imshow(imrgb)
                        plt.title('T = {}'.format(t))
                        plt.pause(0.1)
                        valid_vec[p] = input('{}: Is seg for frame {}_{} valid? '.format(dataset, t, p)) == ''

                row = (im_fname, seg_fname, np.all(valid_vec), dist_fname)
            else:
                row = (im_fname, None, None, None)
            seq_metadata['filelist'].append(row)

        else:
            break
        print("Done frame: {}".format(t))
        t += 1
    with open(os.path.join(root_dir, dataset, 'metadata.pickle'), 'wb') as f:
        pickle.dump(seq_metadata, f, pickle.HIGHEST_PROTOCOL)
