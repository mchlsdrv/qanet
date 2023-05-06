import os
import pandas as pd
from tqdm import tqdm
import logging
import time
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_cv.layers import DropBlock2D

from configs.general_configs import COLUMN_NAMES
from utils.aux_funcs import (
    get_objects,
    load_checkpoint,
    info_log,
    print_pretty_message,
    get_data,
    optimize_learning_rate,
    get_optimizer,
    clear_unnecessary_columns,
    get_data_dict,
)
from utils.activations import (
    Swish
)
import warnings

from utils.callbacks import (
    get_callbacks
)
from utils.data_utils import (
    get_data_loaders,
    DataLoader
)

warnings.simplefilter(action='ignore', category=FutureWarning)
DEBUG = False


class RibCage(keras.Model):
    def __init__(self, model_configs: dict, output_dir: pathlib.Path or str, logger: logging.Logger = None):
        super().__init__()
        self.input_image_dims = model_configs.get('input_image_dims')
        self.logger = logger
        self.activation_layer = self._get_activation(configs=model_configs.get('activation'))

        # - Regularization
        self.kernel_regularizer = self._get_kernel_regularizer(configs=model_configs.get('kernel_regularizer'))

        self.output_dir = output_dir
        if isinstance(self.output_dir, str):
            self.output_dir = pathlib.Path(output_dir)
        assert isinstance(
            self.output_dir, pathlib.Path), \
            f'The save_dir parameter must be of types str or ' \
            f'pathlib.Path, but {type(output_dir)} was provided!'
        if not self.output_dir.is_dir():
            os.makedirs(self.output_dir)

        # - Open the models' configurations file
        self.architecture = model_configs.get('architecture')
        assert isinstance(self.architecture, dict), 'No architecture was chosen!'

        # - Build the model
        self.model = self.build_model()

        # - Train epoch history
        self.train_epch_losses = np.array([])
        self.train_epch_gt_seg_msrs = np.array([])
        self.train_epch_pred_seg_msrs = np.array([])
        self.train_pearson_rs = np.array([])
        self.train_mses = np.array([])
        self.train_btch_smpl_dict = dict()
        self.train_btch_outlier_smpl_dict = dict()

        # - Validation epoch history
        self.val_epch_losses = np.array([])
        self.val_epch_gt_seg_msrs = np.array([])
        self.val_epch_pred_seg_msrs = np.array([])
        self.val_pearson_rs = np.array([])
        self.val_mses = np.array([])
        self.val_btch_smpl_dict = dict()
        self.val_btch_outlier_smpl_dict = dict()

    @staticmethod
    def _get_activation(configs: dict):
        activation = None
        if configs.get('type') == 'swish':
            activation = Swish()
        elif configs.get('type') == 'relu':
            activation = tf.keras.layers.ReLU(
                max_value=None if configs.get('max_value') == 'None'
                else configs.get('max_value'),
                negative_slope=configs.get('negative_slope'),
                threshold=configs.get('threshold'))
        elif configs.get('type') == 'leaky_relu':
            activation = tf.keras.layers.LeakyReLU(alpha=configs.get('alpha'))
        return activation

    @staticmethod
    def _get_kernel_regularizer(configs: dict):
        kernel_regularizer = None
        if configs.get('type') == 'l1':
            kernel_regularizer = tf.keras.regularizers.L1(l1=configs.get('l1'))
        elif configs.get('type') == 'l2':
            kernel_regularizer = tf.keras.regularizers.L2(l2=configs.get('l2'))
        elif configs.get('type') == 'l1l2':
            kernel_regularizer = tf.keras.regularizers.L2(l1=configs.get('l1'),
                                                          l2=configs.get('l2'))
        elif configs.get('type') == 'orthogonal':
            kernel_regularizer = tf.keras.regularizers.OrthogonalRegularizer(
                factor=configs.get('factor'), l2=configs.get('mode'))
        return kernel_regularizer

    def _build_conv2d_block(self, filters: int, kernel_size: int, dropblock_rate: float = 0.0, dropblock_size: int = 7,
                            last: bool = False):
        blk = [
                tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same',
                                       kernel_regularizer=self.kernel_regularizer),
                tf.keras.layers.BatchNormalization(),
                self.activation_layer,
                tf.keras.layers.MaxPool2D(padding='same'),
            ]

        if last and dropblock_rate > 0.0:
            blk.append(DropBlock2D(rate=dropblock_rate, block_size=dropblock_size))

        return keras.Sequential(blk)

    def _build_fully_connected_block(self, units: int, drop_rate: float = 0.0, last: bool = False):
        blck = [
                tf.keras.layers.Dense(units=units, kernel_regularizer=self.kernel_regularizer),
                tf.keras.layers.BatchNormalization(),
            ]

        if not last:           # <= If it's not the last layer - add activation layer
            blck.append(self.activation_layer)

        elif drop_rate > 0.0:  # <= If it's the last layer, and the drop_rate is positive - add Dropout layer
            blck.append(tf.keras.layers.Dropout(rate=drop_rate))

        return keras.Sequential(blck)

    def build_model(self):
        block_filters, block_kernel_sizes = \
            self.architecture.get('conv2d_blocks')['out_channels'], \
            self.architecture.get('conv2d_blocks')['kernel_sizes']

        input_left_rib = tmp_input_left_rib = keras.Input(
            self.input_image_dims + (1,), name='input_left_rib')
        input_right_rib = tmp_input_right_rib = keras.Input(
            self.input_image_dims + (1,), name='input_right_rib')
        input_spine = keras.layers.Concatenate()([
            input_left_rib, input_right_rib])
        fltrs_krnls_lst = list(zip(block_filters, block_kernel_sizes))
        for idx, (filters, kernel_size) in enumerate(fltrs_krnls_lst):
            tmp_input_left_rib = self._build_conv2d_block(
                filters=filters, kernel_size=kernel_size,
                dropblock_rate=self.architecture.get('conv2d_blocks')['dropblock_rate'],
                dropblock_size=self.architecture.get('conv2d_blocks')['dropblock_size'],
                last=idx == len(fltrs_krnls_lst) - 1)(tmp_input_left_rib)
            tmp_input_right_rib = self._build_conv2d_block(
                filters=filters, kernel_size=kernel_size,
                dropblock_rate=self.architecture.get('conv2d_blocks')['dropblock_rate'],
                dropblock_size=self.architecture.get('conv2d_blocks')['dropblock_size'],
                last=idx == len(fltrs_krnls_lst) - 1)(tmp_input_right_rib)
            input_spine = keras.layers.Concatenate()(
                [
                    tmp_input_left_rib,
                    tmp_input_right_rib,
                    self._build_conv2d_block(
                        filters=filters,
                        kernel_size=kernel_size,
                        dropblock_rate=self.architecture.get('conv2d_blocks')['dropblock_rate'],
                        dropblock_size=self.architecture.get('conv2d_blocks')['dropblock_size'],
                        last=idx == len(fltrs_krnls_lst) - 1)(input_spine)
                ]
            )

        layer_units = self.architecture.get('fc_blocks')['out_features']
        fc_layer = keras.layers.Flatten()(input_spine)
        sub_model = None
        for idx, units in enumerate(layer_units):
            sub_model = self._build_fully_connected_block(
                units=units,
                drop_rate=self.architecture.get('fc_blocks')['drop_rate'],
                last=idx == len(layer_units) - 1)
            fc_layer = sub_model(fc_layer)
        output_layer = fc_layer

        return keras.Model(inputs=[input_left_rib, input_right_rib], outputs=[output_layer])

    def call(self, inputs, training: bool = False, **kwargs):
        return self.model(inputs)

    def save(self, save_path: pathlib.Path, **kwargs):
        self.model.save(save_path)

    def summary(self, **kwargs):
        return self.model.summary()

    def _log(self, images, masks, true_seg_measures, pred_seg_measures, training: bool = True):
        with tf.device('CPU:0'):
            # --------------------------------------------------------------
            # - ADD THE HISTORY OF THE TRUE AND THE PREDICTED SEG MEASURES -
            # --------------------------------------------------------------
            if training:
                # - Add the target seg measures to epoch history
                self.train_epch_gt_seg_msrs = np.append(
                    self.train_epch_gt_seg_msrs, true_seg_measures)

                # - Add the modified seg measures to epoch history
                self.train_epch_pred_seg_msrs = np.append(
                    self.train_epch_pred_seg_msrs, pred_seg_measures)

                rnd_smpl_idx = np.random.randint(0, len(images) - 1)

                img = images[rnd_smpl_idx]
                msk = masks[rnd_smpl_idx]
                true_sm = true_seg_measures[rnd_smpl_idx]
                pred_sm = pred_seg_measures[rnd_smpl_idx]
                if true_sm - pred_sm > 0.5:
                    self.train_btch_outlier_smpl_dict = dict(
                        image=img, mask=msk,
                        true_seg_measure=true_sm, pred_seg_measure=pred_sm)
                self.train_btch_smpl_dict = dict(
                    image=img, mask=msk,
                    true_seg_measure=true_sm, pred_seg_measure=pred_sm)

            else:
                # - Add the target seg measures to epoch history
                self.val_epch_gt_seg_msrs = np.append(
                    self.val_epch_gt_seg_msrs, true_seg_measures)

                # - Add the modified seg measures to epoch history
                self.val_epch_pred_seg_msrs = np.append(
                    self.val_epch_pred_seg_msrs, pred_seg_measures)

                rnd_smpl_idx = np.random.randint(0, len(images) - 1)

                img = images[rnd_smpl_idx]
                msk = masks[rnd_smpl_idx]
                true_sm = true_seg_measures[rnd_smpl_idx]
                pred_sm = pred_seg_measures[rnd_smpl_idx]
                if true_sm - pred_sm > 0.5:
                    self.val_btch_outlier_smpl_dict = dict(
                        image=img, mask=msk,
                        true_seg_measure=true_sm, pred_seg_measure=pred_sm)
                self.val_btch_smpl_dict = dict(
                    image=img, mask=msk,
                    true_seg_measure=true_sm, pred_seg_measure=pred_sm)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None],
                                                dtype=tf.float32, name='btch_imgs_aug'),
                                  tf.TensorSpec(shape=[None, None, None],
                                                dtype=tf.float32, name='btch_msks_aug'),
                                  tf.TensorSpec(shape=[None],
                                                dtype=tf.float32, name='btch_true_seg_msrs')
                                  ])
    def learn(self, btch_imgs_aug, btch_msks_aug, btch_true_seg_msrs) -> dict:
        print(f'\n- Train Tracing')
        # - Compute the loss according to the predictions
        with tf.GradientTape() as tape:
            btch_pred_seg_msrs = self.model([btch_imgs_aug, btch_msks_aug], training=True)
            loss = self.compiled_loss(btch_true_seg_msrs, btch_pred_seg_msrs)

        # - Get the weights to adjust according to the loss calculated
        trainable_vars = self.trainable_variables

        # - Calculate gradients
        gradients = tape.gradient(loss, trainable_vars)

        # - Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return dict(loss=loss, batch_seg_mesures=btch_pred_seg_msrs)

    def train_step(self, data) -> dict:

        (btch_imgs_aug, btch_msks_aug), btch_true_seg_msrs = data
        learn_res = self.learn(btch_imgs_aug, btch_msks_aug, btch_true_seg_msrs)
        loss, btch_pred_seg_msrs = learn_res.get('loss'), learn_res.get('batch_seg_mesures')

        (btch_imgs_aug, btch_msks_aug), btch_true_seg_msrs = data

        # Logs
        self._log(
            images=btch_imgs_aug.numpy(),
            masks=btch_msks_aug.numpy(),
            true_seg_measures=btch_true_seg_msrs.numpy(),
            pred_seg_measures=btch_pred_seg_msrs.numpy()[:, 0],
            training=True
        )
        self.train_epch_losses = np.append(self.train_epch_losses, loss.numpy())
        # - Return the mapping metric names to current value
        return {metric.name: metric.result() for metric in self.metrics}

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name='images'),
                                  tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name='masks'),
                                  tf.TensorSpec(shape=[None], dtype=tf.float32, name='seg_scores')
                                  ])
    def validate(self, btch_imgs_aug, btch_msks_aug, btch_true_seg_msrs) -> dict:
        print(f'\n- Validation Tracing')
        # - Compute the loss according to the predictions
        btch_pred_seg_msrs = self.model([btch_imgs_aug, btch_msks_aug], training=False)
        loss = self.compiled_loss(btch_true_seg_msrs, btch_pred_seg_msrs)

        return dict(loss=loss, batch_seg_mesures=btch_pred_seg_msrs)

    def test_step(self, data) -> dict:
        # print('len(data): ', len(data))
        (btch_imgs_aug, btch_msks_aug), btch_true_seg_msrs = data
        val_res = self.validate(btch_imgs_aug, btch_msks_aug, btch_true_seg_msrs)

        loss, btch_pred_seg_msrs = val_res.get('loss'), val_res.get('batch_seg_mesures')
        (btch_imgs_aug, btch_msks_aug), btch_true_seg_msrs = data

        self._log(
            images=btch_imgs_aug.numpy(),
            masks=btch_msks_aug.numpy(),
            true_seg_measures=btch_true_seg_msrs.numpy(),
            pred_seg_measures=btch_pred_seg_msrs.numpy()[:, 0],
            training=False
        )
        self.val_epch_losses = np.append(self.val_epch_losses, loss.numpy())

        return {metric.name: metric.result() for metric in self.metrics}

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name='image'),
                                  tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name='mask'),
                                  ])
    def get_preds(self, image, mask):
        return self.model([image, mask], training=False)

    def infer(self, data_loader) -> float:
        pred_seg_scr_df = self.test(data_loader=data_loader)
        mean_pred = pred_seg_scr_df.loc[:, 'seg_score'].values.mean()
        return mean_pred

    def test(self, data_loader) -> pd.DataFrame:
        t_strt = time.time()

        results_df = pd.DataFrame(columns=COLUMN_NAMES)
        crp_h, crp_w = data_loader.crop_height, data_loader.crop_width
        # - Get the data of the current epoch
        pbar = tqdm(data_loader)
        ptch_pred_mean_seg_scrs = np.array([])
        for idx, (img, msk, img_fl) in enumerate(pbar):
            img_ptchs, msk_ptchs = get_objects(image=img, mask=msk, crop_height=crp_h, crop_width=crp_w)
            # - Get the predictions
            # img_ptchs, msk_ptchs = \
            #     patchify(img, (crp_h, crp_w), step=crp_w).reshape(-1, crp_h, crp_w), \
            #     patchify(msk, (crp_h, crp_w), step=crp_w).reshape(-1, crp_h, crp_w)

            img_ptchs, msk_ptchs = \
                tf.convert_to_tensor(img_ptchs, dtype=tf.float32), \
                tf.convert_to_tensor(msk_ptchs, dtype=tf.float32)

            ptch_pred_mean_seg_scr = self.get_preds(img_ptchs, msk_ptchs).numpy().flatten().mean()

            # - Append the predicted seg measures to the results
            results_df = results_df.append(
                dict(
                    image_file=str(img_fl),
                    gt_mask_file=None,
                    pred_mask_file=str(msk),
                    seg_score=ptch_pred_mean_seg_scr
                ), ignore_index=True)

            pbar.set_postfix(seg=f'{ptch_pred_mean_seg_scr:.3f}')

        return results_df


def get_model(mode: str, hyper_parameters: dict, output_dir: pathlib.Path or str, logger: logging.Logger = None):
    weights_loaded = False

    model_configs = dict(
        input_image_dims=(hyper_parameters.get('augmentations')['crop_height'],
                          hyper_parameters.get('augmentations')['crop_width']),
        architecture=hyper_parameters.get('model')['architecture'],
        kernel_regularizer=dict(
            type=hyper_parameters.get('regularization')[
                'kernel_regularizer_type'],
            l1=hyper_parameters.get('regularization')['kernel_regularizer_l1'],
            l2=hyper_parameters.get('regularization')['kernel_regularizer_l2'],
            factor=hyper_parameters.get('regularization')[
                'kernel_regularizer_factor'],
            mode=hyper_parameters.get('regularization')[
                'kernel_regularizer_mode']
        ),
        activation=dict(
            type=hyper_parameters.get('model')['activation'],
            max_value=hyper_parameters.get('model')[
                'activation_relu_max_value'],
            negative_slope=hyper_parameters.get('model')[
                'activation_relu_negative_slope'],
            threshold=hyper_parameters.get('model')[
                'activation_relu_threshold'],
            alpha=hyper_parameters.get('model')['activation_leaky_relu_alpha']
        )
    )
    model = RibCage(model_configs=model_configs, output_dir=output_dir, logger=logger)
    ckpt_file = hyper_parameters.get(mode)['checkpoint_file']
    weights_loaded = load_checkpoint(model=model, checkpoint_file=ckpt_file)

    if isinstance(logger, logging.Logger):
        info_log(logger=logger, message=model.summary())
    else:
        print(model.summary())

    return model, weights_loaded

def train_model(hyper_parameters: dict, output_dir: pathlib.Path or str, logger: logging.Logger = None):
    # - Load the data
    data_dict = get_data(mode='training', hyper_parameters=hyper_parameters, logger=logger)
    hyper_parameters.get('training')['train_data_len'] = len(data_dict)
    print_pretty_message(
        message=f'Training on {len(data_dict)} examples',
        delimiter_symbol='='
    )

    # MODEL
    # -1- Build the model and optionally load the weights
    model, weights_loaded = get_model(
        mode='training',
        hyper_parameters=hyper_parameters,
        output_dir=output_dir, logger=logger
    )

    # - Get the train and the validation data loaders
    train_dl, val_dl = get_data_loaders(mode='training', data_dict=data_dict,
                                        hyper_parameters=hyper_parameters,
                                        logger=logger)

    # - Get the callbacks and optionally the thread which runs the tensorboard
    callbacks, tb_prc = get_callbacks(callback_type='training',
                                      hyper_parameters=hyper_parameters,
                                      output_dir=output_dir, logger=logger)

    # - If the setting is to launch the tensorboard process automatically
    if tb_prc is not None and hyper_parameters.get('callbacks')['tensorboard_launch']:
        tb_prc.start()

    # - Optimize the learning rate
    if hyper_parameters.get('training')['learning_rate_optimization_epochs'] > 0:
        if hyper_parameters.get('training')['learning_rate_scheduler'] == 'cyclical':
            opt_lr_min, opt_lr_max = optimize_learning_rate(
                model=model,
                data_loader=train_dl,
                learning_rate_min=hyper_parameters.get('training')['learning_rate_scheduler_cyclical_init_lr'],
                learning_rate_max=hyper_parameters.get('training')['learning_rate_scheduler_cyclical_max_lr'],
                epochs=hyper_parameters.get('training')['learning_rate_optimization_epochs'],
                plot_file=output_dir / 'learning_rate_optimization_plot.png'
            )

            # - Assign the optimal learning rate to the initial learning rate in the cyclical lr scheduler
            hyper_parameters.get('training')['learning_rate_scheduler_cyclical_init_lr'] = opt_lr_min
            hyper_parameters.get('training')['learning_rate_scheduler_cyclical_max_lr'] = opt_lr_max

            print(f''''
            Learning Rate Optimization Stats:
                - Optimal initial learning rate: {opt_lr_min}
                - Optimal maximal learning rate: {opt_lr_max}'
            ''')
        else:
            opt_lr, _ = optimize_learning_rate(
                model=model,
                data_loader=train_dl,
                learning_rate_min=hyper_parameters.get('training')['learning_rate_optimization_min'],
                learning_rate_max=hyper_parameters.get('training')['learning_rate_optimization_max'],
                epochs=hyper_parameters.get('training')['learning_rate_optimization_epochs'],
                plot_file=output_dir / 'learning_rate_optimization_plot.png'
            )

            # - Set the optimal learning rate
            hyper_parameters.get('training')['learning_rate'] = opt_lr

            print(f''''
            Learning Rate Optimization Stats:
                - Optimal learning rate: {opt_lr}')
            ''')

    # - Get the optimizer
    if hyper_parameters.get('training')['learning_rate_scheduler'] == 'cyclical':
        hyper_parameters.get('training')['learning_rate_scheduler_cyclical_step_size'] = 2 * len(train_dl)

    optimizer = get_optimizer(args=hyper_parameters)

    # - Compile the model
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=optimizer,
        run_eagerly=True,
        metrics=hyper_parameters.get('training')['metrics']
    )

    # - Train -
    model.fit(
        train_dl,
        batch_size=hyper_parameters.get('training')['batch_size'],
        validation_data=val_dl,
        shuffle=True,
        epochs=hyper_parameters.get('training')['epochs'],
        callbacks=callbacks
    )

    # -> If the setting is to launch the tensorboard process automatically
    if tb_prc is not None and hyper_parameters.get('callbacks')['tensorboard_launch']:
        tb_prc.join()

    return model


def test_model(model, hyper_parameters: dict, output_dir: pathlib.Path or str, logger: logging.Logger = None):
    test_res_df = None
    df_fl = pathlib.Path(hyper_parameters.get('test')['dataframe_file'])
    if df_fl.is_file():
        # - Load the dataframe
        test_res_df = pd.read_csv(df_fl)

        # - Clear unnecessary columns
        test_res_df = clear_unnecessary_columns(dataframe=test_res_df)

        # - Clear the nans
        test_res_df = test_res_df.loc[~test_res_df.loc[:, 'seg_score'].isna()].reset_index(drop=True)

        # - Create the data tuples for the to be fed into the get_data_dict
        data_file_tuples = [(img_fl, pred_msk_fl) for
                            img_fl, _, pred_msk_fl, _ in test_res_df.values]

        # - Construct the data dictionary containing the image files, images, mask files and masks
        data_dict = get_data_dict(image_mask_file_tuples=data_file_tuples)

        test_dl = DataLoader(
            mode='test',
            data_dict=data_dict,
            file_keys=list(data_dict.keys()),
            crop_height=hyper_parameters.get('augmentations')['crop_height'],
            crop_width=hyper_parameters.get('augmentations')['crop_width'],
            batch_size=1,
            logger=logger
        )

        # MODEL
        # -1- Build the model and optionally load the weights
        if model is None:
            model, weights_loaded = get_model(
                mode='test',
                hyper_parameters=hyper_parameters,
                output_dir=output_dir, logger=logger)

            chkpt_dir = hyper_parameters.get("test")["checkpoint_dir"]
            assert weights_loaded, f'Could not load weights from {pathlib.Path(chkpt_dir)}!'

        # - Infer
        pred_df = model.test(data_loader=test_dl)

        test_res_df.loc[test_res_df.loc[:, 'image_file'].isin(pred_df.loc[:, 'image_file']), 'pred_seg_score'] = \
            pred_df.loc[:, 'seg_score']

        print_pretty_message(
            message=f'Testing {len(data_dict)} images'
        )

    return test_res_df


def infer_data(model, hyper_parameters: dict, output_dir: pathlib.Path or str, logger: logging.Logger = None):
    # - Load the data
    data_dict = get_data(mode='inference', hyper_parameters=hyper_parameters, logger=logger)

    inf_dl = DataLoader(
        mode='inference',
        data_dict=data_dict,
        file_keys=list(data_dict.keys()),
        crop_height=hyper_parameters.get('augmentations')['crop_height'],
        crop_width=hyper_parameters.get('augmentations')['crop_width'],
        batch_size=1,
        logger=logger
    )

    print_pretty_message(
        message=f'Inferring {len(data_dict)} examples',
        delimiter_symbol='='
    )

    # - Infer
    prediction = model.infer(data_loader=inf_dl)

    return prediction