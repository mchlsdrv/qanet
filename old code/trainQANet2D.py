import argparse
import os
import pickle
import tensorflow as tf
import Networks as Nets
import Params
from distutils.util import strtobool
import DataHandeling
import sys
import io
from utils import log_print
import matplotlib as mpl

mpl.use('Agg')  # No display
import matplotlib.pyplot as plt

__author__ = 'arbellea@post.bgu.ac.il'

try:
    import tensorflow.python.keras as k
except AttributeError:
    import tensorflow.keras as k

if not tf.__version__.split('.')[0] == '2':
    raise ImportError('Required tensorflow version 2.x. current version is: {}'.format(tf.__version__))


def train():
    # Data input
    train_data_provider = params.train_data_provider
    val_data_provider = params.val_data_provider
    coord = tf.train.Coordinator()
    train_data_provider.start_queues(coord)
    val_data_provider.start_queues(coord)

    # Model
    device = '/gpu:0' if int(params.gpu_id) >= 0 else '/cpu:0'
    with tf.device(device):
        input_shape_left = (
            params.crop_size + (train_data_provider.image_channel_depth,
                                ) if params.channel_axis == 3 else (train_data_provider.image_channel_depth,
                                                                    ) + params.crop_size)
        input_shape_right = params.crop_size + (1,) if params.channel_axis == 3 else (1,) + params.crop_size
        params.net_kernel_params['input_shape_left'] = input_shape_left
        params.net_kernel_params['input_shape_right'] = input_shape_right
        model = params.net_model().build(input_shape_left, input_shape_right, params.net_kernel_params,
                                         params.data_format)

        # Losses and Metrics

        mse_loss = k.losses.MeanSquaredError()
        train_loss = k.metrics.Mean(name='train_loss')

        val_loss = k.metrics.Mean(name='val_loss')

        # Save Checkpoints
        optimizer = k.optimizers.Adam(lr=params.learning_rate)
        ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), optimizer=optimizer, net=model)
        if params.load_checkpoint:

            if os.path.isdir(params.load_checkpoint_path):
                latest_checkpoint = tf.train.latest_checkpoint(params.load_checkpoint_path)
            else:
                latest_checkpoint = params.load_checkpoint_path
            try:
                ckpt.restore(latest_checkpoint)
                log_print("Restored from {}".format(latest_checkpoint))
            except tf.errors.NotFoundError:
                raise ValueError("Could not load checkpoint: {}".format(latest_checkpoint))

        else:
            log_print("Initializing from scratch.")

        manager = tf.train.CheckpointManager(ckpt, os.path.join(params.experiment_save_dir, 'tf_ckpts'),
                                             max_to_keep=params.save_checkpoint_max_to_keep,
                                             keep_checkpoint_every_n_hours=params.save_checkpoint_every_N_hours)

        @tf.function
        def train_step(image, mod_seg, label):
            with tf.GradientTape() as tape:
                predictions = model([image, mod_seg], True)
                loss = mse_loss(label, tf.squeeze(predictions))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            ckpt.step.assign_add(1)
            train_loss(loss)
            predictions = tf.maximum(0., tf.minimum(1., predictions))

            return predictions, loss

        @tf.function
        def train10(train_data_provider):
            image_batch = seg_batch = modified_seg_batch = target_jaccard = fnames_batch = train_pred = None
            for _ in range(10):
                (image_batch, seg_batch, modified_seg_batch, target_jaccard,
                 fnames_batch) = train_data_provider.get_batch()
                train_pred, _ = train_step(image_batch, modified_seg_batch, target_jaccard)
            return image_batch, seg_batch, modified_seg_batch, target_jaccard, fnames_batch, train_pred

        @tf.function
        def val_step(image, mod_seg, label):
            predictions = model([image, mod_seg], False)
            loss = mse_loss(label, tf.squeeze(predictions))

            val_loss(loss)
            predictions = tf.maximum(0., tf.minimum(1., predictions))

            return predictions, loss

        train_summary_writer = val_summary_writer = train_scalars_dict = val_scalars_dict = None
        if not params.dry_run:
            train_log_dir = os.path.join(params.experiment_log_dir, 'train')
            val_log_dir = os.path.join(params.experiment_log_dir, 'val')
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

            val_summary_writer = tf.summary.create_file_writer(val_log_dir)
            train_scalars_dict = {'Loss': train_loss, 'Q Stat': train_data_provider.q_stat}
            val_scalars_dict = {'Loss': val_loss, 'Q Stat': train_data_provider.q_stat}

        def tf_scatter(gt, predictions):
            with tf.device('/cpu:0'):
                plt.figure()

                plt.scatter(gt, predictions)
                plt.scatter(gt[0:1], predictions[0:1], c='r', marker='*')
                plt.xlim((0, 1))
                plt.ylim((0, 1))
                plt.plot([0, 1], [0, 1], '--')
                plt.xlabel('GT Jaccard')
                plt.ylabel('Predicted Jaccard')
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                image_str = buf.getvalue()
                tf_image = tf.image.decode_png(image_str, channels=3)
                tf_image = tf.expand_dims(tf_image, 0)
                if params.channel_axis == 1:
                    tf_image = tf.transpose(tf_image, (0, 3, 1, 2))
            return tf_image

        def tboard(writer, step, scalar_loss_dict, images_dict):

            with writer.as_default():
                with tf.device('/cpu:0'):
                    for scalar_loss_name, scalar_loss in scalar_loss_dict.items():
                        try:
                            tf.summary.scalar(scalar_loss_name, scalar_loss.result(), step=step)
                        except AttributeError:
                            tf.summary.scalar(scalar_loss_name, scalar_loss(), step=step)
                    for image_name, image in images_dict.items():
                        if params.channel_axis == 1:
                            image = tf.transpose(image, (0, 2, 3, 1))
                        tf.summary.image(image_name, image, max_outputs=1, step=step)

        template = '{}: Step {}, Loss: {}'
        try:
            # if True:
            if not params.dry_run:
                log_print('Saving Model of inference:')
                model_fname = os.path.join(params.experiment_save_dir, 'model.ckpt'.format(int(ckpt.step)))
                model.save_weights(model_fname, save_format='tf')
                with open(os.path.join(params.experiment_save_dir, 'model_params.pickle'), 'wb') as fobj:
                    pickle.dump({'name': model.__class__.__name__, 'params': (input_shape_left, input_shape_right,
                                                                              params.net_kernel_params,)},
                                fobj, protocol=pickle.HIGHEST_PROTOCOL)
                log_print('Saved Model to file: {}'.format(model_fname))

            for _ in range(int(ckpt.step), params.num_iterations + 1):
                (image_batch, seg_batch, modified_seg_batch, target_jaccard,
                 fnames_batch, train_pred) = train10(train_data_provider)
                modified_seg_onehot = tf.one_hot(
                    tf.cast(tf.squeeze(modified_seg_batch, params.channel_axis), tf.int32),
                    depth=3)
                seg_onehot = tf.one_hot(tf.cast(tf.squeeze(seg_batch, params.channel_axis), tf.int32),
                                        depth=3)
                if params.channel_axis == 1:
                    seg_onehot = tf.transpose(seg_onehot, (0, 3, 1, 2))
                    modified_seg_onehot = tf.transpose(modified_seg_onehot, (0, 3, 1, 2))
                if params.profile:
                    tf.summary.trace_on(graph=True, profiler=True)
                # train_pred, _ = train_step(image_batch, modified_seg_batch, target_jaccard)
                if params.profile:
                    with train_summary_writer.as_default():
                        tf.summary.trace_export('train_step', step=int(ckpt.step),
                                                profiler_outdir=params.experiment_log_dir)

                if not int(ckpt.step) % params.write_to_tb_interval:
                    if not params.dry_run:
                        train_scatter = tf_scatter(target_jaccard, train_pred)
                        display_image = image_batch
                        display_image = display_image - tf.reduce_min(display_image, axis=(1, 2, 3), keepdims=True)
                        display_image = display_image / tf.reduce_max(display_image, axis=(1, 2, 3), keepdims=True)
                        train_imgs_dict = {'Image': display_image, 'GT': seg_onehot,
                                           'Segmentation': modified_seg_onehot,
                                           'Scatter': train_scatter}
                        tboard(train_summary_writer, int(ckpt.step), train_scalars_dict, train_imgs_dict)
                        log_print('Printed Training Step: {} to Tensorboard'.format(int(ckpt.step)))
                    else:
                        log_print("WARNING: dry_run flag is ON! Not saving checkpoints or tensorboard data")

                if int(ckpt.step) % params.save_checkpoint_iteration == 0 or int(ckpt.step) == params.num_iterations:
                    if not params.dry_run:
                        save_path = manager.save(int(ckpt.step))
                        model.save_weights(model_fname, save_format='tf')
                        log_print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

                    else:
                        log_print("WARNING: dry_run flag is ON! Mot saving checkpoints or tensorboard data")
                if not int(ckpt.step) % params.print_to_console_interval:
                    log_print(template.format('Training', int(ckpt.step),
                                              train_loss.result()))

                if not int(ckpt.step) % params.validation_interval:

                    (val_image_batch, val_seg_batch, val_mod_seg_batch, val_target_jaccard,
                     val_fnames_batch) = val_data_provider.get_batch()

                    seg_onehot = tf.one_hot(tf.cast(tf.squeeze(val_seg_batch, params.channel_axis), tf.int32),
                                            depth=3)
                    val_mod_seg_onehot = tf.one_hot(
                        tf.cast(tf.squeeze(val_mod_seg_batch, params.channel_axis), tf.int32),
                        depth=3)
                    if params.channel_axis == 1:
                        seg_onehot = tf.transpose(seg_onehot, (0, 3, 1, 2))
                        val_mod_seg_onehot = tf.transpose(val_mod_seg_onehot, (0, 3, 1, 2))

                    val_pred, _ = val_step(val_image_batch, val_mod_seg_batch, val_target_jaccard)

                    if not params.dry_run:
                        val_scatter = tf_scatter(val_target_jaccard, val_pred)
                        display_image = val_image_batch
                        display_image = display_image - tf.reduce_min(display_image, axis=(1, 2, 3), keepdims=True)
                        display_image = display_image / tf.reduce_max(display_image, axis=(1, 2, 3), keepdims=True)
                        val_imgs_dict = {'Image': display_image, 'GT': seg_onehot,
                                         'Segmentation': val_mod_seg_onehot,
                                         'Scatter': val_scatter}
                        tboard(val_summary_writer, int(ckpt.step), val_scalars_dict, val_imgs_dict)
                        log_print('Printed Validation Step: {} to Tensorboard'.format(int(ckpt.step)))
                    else:
                        log_print("WARNING: dry_run flag is ON! Not saving checkpoints or tensorboard data")

                    log_print(template.format('Validation', int(ckpt.step),
                                              val_loss.result()))

        except KeyboardInterrupt as err:
            if not params.dry_run:
                log_print('Saving Model Before closing due to error: {}'.format(str(err)))
                save_path = manager.save(int(ckpt.step))
                log_print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                # raise err

        except Exception as err:
            #
            raise err
        finally:
            if not params.dry_run:
                log_print('Saving Model of inference:')
                model_fname = os.path.join(params.experiment_save_dir, 'model.ckpt'.format(int(ckpt.step)))
                model.save_weights(model_fname, save_format='tf')
                with open(os.path.join(params.experiment_save_dir, 'model_params.pickle'), 'wb') as fobj:
                    pickle.dump({'name': model.__class__.__name__, 'params': (params.net_kernel_params,)},
                                fobj, protocol=pickle.HIGHEST_PROTOCOL)
                log_print('Saved Model to file: {}'.format(model_fname))
            else:
                log_print('WARNING: dry_run flag is ON! Not Saving Model')
            log_print('Closing gracefully')
            coord.request_stop()
            coord.join()
            log_print('Done')


if __name__ == '__main__':

    class AddNets(argparse.Action):
        import Networks as Nets

        def __init__(self, option_strings, dest, **kwargs):
            super(AddNets, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            nets = [getattr(Nets, v) for v in values]
            # noinspection PyUnresolvedReferences
            setattr(namespace, self.dest, nets)


    # noinspection PyUnresolvedReferences
    class AddReader(argparse.Action):

        def __init__(self, option_strings, dest, **kwargs):
            super(AddReader, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            reader = getattr(DataHandeling, values)
            setattr(namespace, self.dest, reader)


    # noinspection PyUnresolvedReferences
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


    arg_parser = argparse.ArgumentParser(description='Run Train LSTMUnet Segmentation')
    arg_parser.add_argument('-n', '--experiment_name', dest='experiment_name', type=str,
                            help="Name of experiment")
    arg_parser.add_argument('--gpu_id', dest='gpu_id', type=str,
                            help="Visible GPUs: example, '0,2,3'")
    arg_parser.add_argument('--dry_run', dest='dry_run', action='store_const', const=True,
                            help="Do not write any outputs: for debugging only")
    arg_parser.add_argument('--profile', dest='profile', type=bool,
                            help="Write profiling data to tensorboard. For debugging only")
    arg_parser.add_argument('--root_data_dir', dest='root_data_dir', type=str,
                            help="Root folder containing training data")
    arg_parser.add_argument('--data_provider_class', dest='data_provider_class', type=str, action=AddReader,
                            help="Type of data provider")
    arg_parser.add_argument('--dataset', dest='dataset', type=str, action=AddDatasets, nargs='+',
                            help="Datasets to run. string of pairs: DatasetName, SequenceNumber")
    arg_parser.add_argument('--val_dataset', dest='val_dataset', type=str, action=AddDatasets, nargs='+',
                            help="Datasets to run. string of pairs DatasetName, SequenceNumber")
    arg_parser.add_argument('--net_gpus', dest='net_gpus', type=int, nargs='+',
                            help="gpus for each net: example: 0 0 1")
    arg_parser.add_argument('--net_types', dest='net_types', type=int, nargs='+', action=AddNets,
                            help="Type of nets")
    arg_parser.add_argument('--crop_size', dest='crop_size', type=int, nargs=2,
                            help="crop size for y and x dimensions: example: 160 160")
    arg_parser.add_argument('--train_q_capacity', dest='train_q_capacity', type=int,
                            help="Capacity of training queue")
    arg_parser.add_argument('--val_q_capacity', dest='val_q_capacity', type=int,
                            help="Capacity of validation queue")
    arg_parser.add_argument('--num_train_threads', dest='num_train_threads', type=int,
                            help="Number of train data threads")
    arg_parser.add_argument('--num_val_threads', dest='num_val_threads', type=int,
                            help="Number of validation data threads")
    arg_parser.add_argument('--data_format', dest='data_format', type=str, choices=['NCHW', 'NWHC'],
                            help="Data format NCHW or NHWC")
    arg_parser.add_argument('--batch_size', dest='batch_size', type=int,
                            help="Batch size")
    arg_parser.add_argument('--unroll_len', dest='unroll_len', type=int,
                            help="LSTM unroll length")
    arg_parser.add_argument('--num_iterations', dest='num_iterations', type=int,
                            help="Maximum number of training iterations")
    arg_parser.add_argument('--validation_interval', dest='validation_interval', type=int,
                            help="Number of iterations between validation iteration")
    arg_parser.add_argument('--load_checkpoint', dest='load_checkpoint', action='store_const', const=True,
                            help="Load from checkpoint")
    arg_parser.add_argument('--load_checkpoint_path', dest='load_checkpoint_path', type=str,
                            help="path to checkpoint, used only with --load_checkpoint")
    arg_parser.add_argument('--continue_run', dest='continue_run', action='store_const', const=True,
                            help="Continue run in existing directory")
    arg_parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                            help="Learning rate")
    arg_parser.add_argument('--class_weights', dest='class_weights', type=float, nargs=3,
                            help="class weights for background, foreground and edge classes")
    arg_parser.add_argument('--save_checkpoint_dir', dest='save_checkpoint_dir', type=str,
                            help="root directory to save checkpoints")
    arg_parser.add_argument('--save_log_dir', dest='save_log_dir', type=str,
                            help="root directory to save tensorboard outputs")
    arg_parser.add_argument('--tb_sub_folder', dest='tb_sub_folder', type=str,
                            help="sub-folder to save outputs")
    arg_parser.add_argument('--save_checkpoint_iteration', dest='save_checkpoint_iteration', type=int,
                            help="number of iterations between save checkpoint")
    arg_parser.add_argument('--save_checkpoint_max_to_keep', dest='save_checkpoint_max_to_keep', type=int,
                            help="max recent checkpoints to keep")
    arg_parser.add_argument('--save_checkpoint_every_N_hours', dest='save_checkpoint_every_N_hours', type=int,
                            help="keep checkpoint every N hours")
    arg_parser.add_argument('--write_to_tb_interval', dest='write_to_tb_interval', type=int,
                            help="Interval between writes to tensorboard")
    sys_args = sys.argv

    input_args = arg_parser.parse_args()
    args_dict = {key: val for key, val in vars(input_args).items() if not (val is None)}
    params = Params.QANetParams(args_dict)
    # params = Params.QANetParamsLSC(args_dict)
    tf_eps = tf.constant(1E-8, name='epsilon')
    # try:
    #     train()
    # finally:
    #     log_print('Done')
    train()
