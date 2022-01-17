from DataHandeling import CTCRAMReaderQANet2D
import os
import utils
import cv2
import numpy as np
import pathlib


def run_main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    root_dir = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Test/'
    # root_dir = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/'
    save_path = '/media/rrtammyfs/Users/michael/QANet/TestData/SIM+V2'
    # save_path = '/media/rrtammyfs/Users/arbellea/QANet/TestData/SIM+V2'
    os.makedirs(save_path, exist_ok=True)
    # sequence_folder_list = [(os.path.join(root_dir, 'Fluo-N2DH-SIM+'), '01'),
    #                         (os.path.join(root_dir, 'Fluo-N2DH-SIM+'), '02')]
    sequence_folder_list = [(os.path.join(root_dir, 'Fluo-N2DH-MYSIMV3'), '{:02d}'.format(s + 1)) for s in range(46, 51)]
    image_crop_size = (256, 256)
    batch_size = 10
    queue_capacity = 250
    num_threads = 2
    data_format = 'NHWC'
    dataset_percent = -0.2
    # dataset_percent = -1
    data = CTCRAMReaderQANet2D(sequence_folder_list, image_crop_size, batch_size, queue_capacity, num_threads,
                               data_format, dataset_percent)

    debug = False
    data.start_queues(debug=debug)
    all_jaccard = []
    for i in range(100):
        utils.log_print(i)
        image_batch, seg_batch, modified_seg_batch, jaccard, fnames = data.get_batch()
        np.save(f'{save_path}/image_batch{i}.npy', image_batch.numpy())
        np.save(f'{save_path}/seg_batch{i}.npy', seg_batch.numpy())
        np.save(f'{save_path}/modified_seg_batch{i}.npy', modified_seg_batch.numpy())
        np.save(f'{save_path}/jaccard{i}.npy', jaccard.numpy())
        np.save(f'{save_path}/fnames{i}.npy', fnames.numpy())
        utils.log_print(i, image_batch.shape, seg_batch.shape, jaccard)
        all_jaccard.append(jaccard)
        pass

if __name__ == '__main__':
    run_main()
