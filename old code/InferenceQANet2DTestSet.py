import argparse
import os
import pickle
import scipy.ndimage
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import Networks as Nets
import Params
from distutils.util import strtobool
import DataHandeling
import sys
from utils import log_print, get_model, bbox_crop, bbox_fill
import RibCage
import matplotlib.pylab as pylab
import seaborn as sns
import pandas as pd
from utils import log_print as print
params = {'legend.fontsize': 'xx-large',
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large',
          }
pylab.rcParams.update(params)

__author__ = 'arbellea@post.bgu.ac.il'

try:
    import tensorflow.python.keras as k
except AttributeError:
    import tensorflow.keras as k
if not tf.__version__.split('.')[0] == '2':
    raise ImportError(f'Required tensorflow version 2.x. current version is: {tf.__version__}')


def plot(array_gt, array_est, folder):
    array_gt = np.array(array_gt)
    array_est = np.array(array_est)
    diff = array_gt - array_est
    data = pd.DataFrame({'x': array_gt, 'y': array_est})
    with sns.axes_style("darkgrid"):
        sns.set_context('paper', font_scale=2)
        g = sns.jointplot(x='x', y='y', data=data, kind='reg', xlim=(0, 1), ylim=(0, 1),
                          joint_kws={'scatter_kws': {'alpha': 0.2}, 'fit_reg': False})
        plt.plot([0, 1], [0, 1])
        g.set_axis_labels('GT Quality Value', 'Estimated Quality Value')
        mse = ((array_gt-array_est)**2).mean()
        print('{}, R Value:{}, MSE: {}'.format(folder, np.corrcoef(array_gt, array_est)[0][1], mse))
        print(folder, array_est.mean(), array_gt.mean())
        plt.figure()
        plt.cla()
        abs_diff_h, abs_diff_bins, _ = plt.hist(np.abs(diff), bins=100, range=(0, 1.))
        abs_diff_h = abs_diff_h/abs_diff_h.sum()
        plt.xlabel('Abs Error')
        plt.ylabel('Freq')
        plt.title(folder)
        plt.figure()
        plt.cla()
        plt.hist(diff, bins=100, range=(-1, 1))
        plt.xlabel('Error')
        plt.ylabel('Freq')
        plt.title(folder)
        plt.figure()
        plt.cla()
        plt.plot(abs_diff_bins[:-1], np.cumsum(abs_diff_h), linewidth=2)
        plt.xlabel('Absolute Error Tolerance')
        plt.ylabel('Hit Rate')
        plt.title(folder)
        plt.ylim((0, 1))

    print(folder, array_est.mean(), array_gt.mean())
    print('Folder: {}: Bias: {}'.format(folder, (array_gt - array_est).mean()))

def inference():
    # Load Model
    with open(os.path.join(params.model_path, 'model_params.pickle'), 'rb') as fobj:
        model_dict = pickle.load(fobj)
    model_cls = RibCage.RibCage

    device = '/gpu:3' #if params.gpu_id >= 0 else '/cpu:0'
    with tf.device(device):
        input_shape_left = model_dict['params'][-1]['input_shape_left']
        input_shape_right = model_dict['params'][-1]['input_shape_right']
        net_params = model_dict['params'][-1]
        model = model_cls().build(input_shape_left, input_shape_right, net_params, data_format=params.data_format)
        if params.override_checkpoint:
            ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), net=model)
            ckpt.restore(params.override_checkpoint)
        else:
            model.load_weights(os.path.join(params.model_path, 'model.ckpt'))
        log_print(
            "Restored from {}".format(params.override_checkpoint or os.path.join(params.model_path, 'model.ckpt')))

    test_data_provider = params.test_data_provider
    coord = tf.train.Coordinator()
    test_data_provider.start_queues(coord)

    dataset = params.data_reader(**params.data_params).dataset
    all_jac_gt_dict = {}
    all_jac_est_dict = {}
    try:
        for t, data in enumerate(dataset):
            print(t)
            if params.has_gt:
                (image, seg, gt, jac_gt, f) = data
            else:
                (image, seg, f) = data
                jac_gt = None
            folder = f.numpy().decode('utf-8')
            if not folder in all_jac_gt_dict.keys():
                all_jac_gt_dict[folder] = []
                all_jac_est_dict[folder] = []
            image_shape = image.shape
            seg_shape = seg.shape
            if len(image_shape) == 2:
                if params.data_format == 'NCHW':
                    image = tf.reshape(image, [1, 1, image_shape[0], image_shape[1]])
                    seg = tf.reshape(seg, [1, 1, seg_shape[0], seg_shape[1]])
                else:
                    image = tf.reshape(image, [1, image_shape[0], image_shape[1], 1])
                    seg = tf.reshape(seg, [1, seg_shape[0], seg_shape[1], 1])
            elif len(image_shape) == 3:
                image = tf.reshape(image, [1, image_shape[0], image_shape[1], image_shape[2]])
                seg = tf.reshape(seg, [1, seg_shape[0], seg_shape[1], seg_shape[2]])
            elif len(image_shape) == 4:
                pass
            else:
                raise ValueError()


            if np.all(seg[0][0][:2,:2].numpy() == np.array([[1., 2.],[2., 2.]])):
                all_jac_est_dict[folder].append(0.)
            else:
                jac_estimate = model((image, seg), training=False)
                all_jac_est_dict[folder].append(
                    np.maximum(np.minimum(jac_estimate.numpy().squeeze().mean() + params.bias,
                                          1.), 0.))
            if params.has_gt:
                all_jac_gt_dict[folder].append(jac_gt.numpy())

    except (KeyboardInterrupt, ValueError) as err:
        print('Error: {}'.format(str(err)))

    except Exception as err:
        raise err
    finally:
        print('Done!')
        mg = []
        me = []
        all_jac_gt_np = []
        all_jac_est_np = []
        if not params.dry_run and params.has_gt:
            for (folder, array_gt), array_est in zip(all_jac_gt_dict.items(), all_jac_est_dict.values()):
                all_jac_gt_np.extend(array_gt)
                all_jac_est_np.extend(array_est)
            plot(all_jac_gt_np, all_jac_est_np, 'All')
            plt.show()


def choose_gpu(gpu_id: int = 0):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            physical_gpus = tf.config.list_physical_devices('GPU')
            # upper_line = ''
            print(f'''
            ====================================================
            > Running on: {physical_gpus}
            ====================================================
            ''')
        except RuntimeError as error:
            print(error)

if __name__ == '__main__':

    choose_gpu(gpu_id=3)
    class AddNets(argparse.Action):
        import Networks as Nets

        def __init__(self, option_strings, dest, **kwargs):
            super(AddNets, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            nets = [getattr(Nets, v) for v in values]
            setattr(namespace, self.dest, nets)


    class AddReader(argparse.Action):

        def __init__(self, option_strings, dest, **kwargs):
            super(AddReader, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            reader = getattr(DataHandeling, values)
            setattr(namespace, self.dest, reader)


    class AddDatasets(argparse.Action):

        def __init__(self, option_strings, dest, *args, **kwargs):

            super(AddDatasets, self).__init__(option_strings, dest, *args, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):

            if len(values) % 2:
                raise ValueError("dataset values should be of length 2*N where N is the number of datasets")
            datastets = []
            for i in range(0, len(values), 2):
                datastets.append((values[i], strtobool(values[i + 1])))
            setattr(namespace, self.dest, datastets)


    arg_parser = argparse.ArgumentParser(description='Run Inference LSTMUnet Segmentation')
    arg_parser.add_argument('--gpu_id', dest='gpu_id', type=str,
                            help="Visible GPUs: example, '0,2,3', use -1 for CPU")
    arg_parser.add_argument('--model_path', dest='model_path', type=str,
                            help="Path to trained model generated by train2D.py, folder should contain model.ckpt.*")

    arg_parser.add_argument('--sequence_path', dest='sequence_path', type=str,
                            help="Path to sequence images. Folder should contain image files")
    arg_parser.add_argument('--filename_format', dest='filename_format', type=str,
                            help="Format of file using wildcard (*) to indicate timestep. Default: 't*.tif'")
    arg_parser.add_argument('--data_format', dest='data_format', type=str, choices=['NCHW', 'NWHC'],
                            help="Data format NCHW or NHWC")

    arg_parser.add_argument('--min_cell_size', dest='min_cell_size', type=int,
                            help="Minimum cell size")
    arg_parser.add_argument('--max_cell_size', dest='max_cell_size', type=int,
                            help="Maximum cell size")
    arg_parser.add_argument('--num_iterations', dest='num_iterations', type=int,
                            help="Maximum number of training iterations")
    arg_parser.add_argument('--edge_dist', dest='edge_dist', type=int,
                            help="Maximum edge width to add to cell object")
    arg_parser.add_argument('--pre_sequence_frames', dest='pre_sequence_frames', type=int,
                            help="Number of frames to run before sequence, uses mirror of first N frames.")
    arg_parser.add_argument('--save_intermediate', dest='save_intermediate', action='store_const', const=True,
                            help="Save intermediate files")
    arg_parser.add_argument('--save_intermediate_path', dest='save_intermediate_path', type=str,
                            help="Path to save intermediate files, used only with --save_intermediate")
    arg_parser.add_argument('--dry_run', dest='dry_run', action='store_const', const=True,
                            help="Do not write any outputs: for debugging only")
    sys_args = sys.argv

    input_args = arg_parser.parse_args()
    args_dict = {key: val for key, val in vars(input_args).items() if not (val is None)}
    params = Params.QANetCTCTestParams(args_dict)
    tf_eps = tf.constant(1E-8, name='epsilon')
    try:
        inference()
    except Exception as err:
        raise err
    finally:
        log_print('Done')
