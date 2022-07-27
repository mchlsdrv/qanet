import argparse
import logging.config
import pickle as pkl
from functools import partial

import albumentations as A
import torchvision
import yaml
from albumentations.pytorch import ToTensorV2
from utils.logging_funcs import info_log, err_log

from configs.general_configs import (
    OUTPUT_DIR,

    EPOCHS,
    BATCH_SIZE,
    OPTIMIZER_LR,

    OPTIMIZER,
    OPTIMIZER_WEIGHT_DECAY,
    OPTIMIZER_MOMENTUM_DECAY,

    KERNEL_REGULARIZER_TYPE,
    KERNEL_REGULARIZER_L1,
    KERNEL_REGULARIZER_L2,
    KERNEL_REGULARIZER_FACTOR,
    KERNEL_REGULARIZER_MODE,

    CHECKPOINT_DIR,
    CHECKPOINT_FILE,
    OPTIMIZER_BETA_1, OPTIMIZER_BETA_2,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    IN_CHANNELS, OUT_CHANNELS,
    OPTIMIZER_RHO, OPTIMIZER_MOMENTUM, OPTIMIZER_DAMPENING, OPTIMIZER_LR_DECAY, OPTIMIZER_EPS,
    NUM_WORKERS, INFERENCE_DIR
)
import logging
import os
import pathlib

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from configs.general_configs import (
    VAL_PROP,
    LOSS,
    TRAIN_DIR,
    EPSILON,
    REDUCE_LR_ON_PLATEAU_MIN,
    REDUCE_LR_ON_PLATEAU,
    EARLY_STOPPING,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_ON_PLATEAU_FACTOR,
    REDUCE_LR_ON_PLATEAU_PATIENCE
)
from custom.models import RibCage

from utils.data_utils import (
    get_data_loaders,
)

from custom.augs import (
    train_augs,
    val_augs, test_augs
)
from configs.general_configs import (
    MODEL_CONFIGS_FILE,
)


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])


def get_accuracy(data_loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0.0
    model.eval()

    with torch.no_grad():
        for img, seg, seg_aug, seg_msr in data_loader:
            x = x.to(device=device)
            y = y.to(device='cpu').unsqueeze(1)

            preds = torch.sigmoid(model(x)).to('cpu')
            preds = (preds > 0.5).float()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            acc = (num_correct / num_pixels)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
    dice_score = dice_score / len(data_loader)
    print(f'Accuracy: {acc * 100 :.4f}%')
    print(f'Dice Score: {dice_score:.4f}')

    model.train()

    return acc, dice_score


def save_preds(data_loader, model, save_dir='./results/images', device='cuda'):
    model.eval()
    for idx, (x, y) in enumerate(data_loader):
        x = x.to(device=device)

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f'{save_dir}/pred_mask_{idx}.png')
        torchvision.utils.save_image(y.unsqueeze(1), f'{save_dir}/gt_mask_{idx}.png')

    model.train()


def read_yaml(data_file: pathlib.Path):
    data = None
    if data_file.is_file():
        with data_file.open(mode='r') as f:
            data = yaml.safe_load(f.read())
    return data


def get_logger(configs_file, save_file):
    logger = None
    try:
        if configs_file.is_file():
            with configs_file.open(mode='r') as f:
                configs = yaml.safe_load(f.read())

                # Assign a valid path to the log file
                configs['handlers']['logfile']['filename'] = str(save_file)
                logging.config.dictConfig(configs)

        logger = logging.getLogger(__name__)
    except Exception as err:
        err_log(logger=logger, message=str(err))

    return logger


def get_model_configs(configs_file: pathlib.Path, logger: logging.Logger):
    if configs_file.is_file():
        model_configs = read_yaml(configs_file)
        if model_configs is not None:
            info_log(logger=logger, message=f'The model configs were successfully loaded from \'{configs_file}\'!')
        else:
            info_log(logger=logger, message=f'The model configs file ({configs_file}) does not contain valid model configurations!')
    else:
        info_log(logger=logger, message=f'The model configs file ({configs_file}) does not exist!')
    return model_configs


def get_runtime(seconds: float):
    hrs = int(seconds // 3600)
    mins = int((seconds - hrs * 3600) // 60)
    sec = seconds - hrs * 3600 - mins * 60

    # - Format the strings
    hrs_str = str(hrs)
    if hrs < 10:
        hrs_str = '0' + hrs_str
    min_str = str(mins)
    if mins < 10:
        min_str = '0' + min_str
    sec_str = f'{sec:.3}'
    if sec < 10:
        sec_str = '0' + sec_str

    return hrs_str + ':' + min_str + ':' + sec_str + '[H:M:S]'


def get_augs(args: dict):
    train_augs = A.Compose([
        A.Resize(height=args.get('image_height'), width=args.get('image_width')),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.CLAHE(p=1.),
        A.ToFloat(p=1.),
        ToTensorV2()
    ])

    val_augs = A.Compose([
        A.Resize(height=args.get('image_height'), width=args.get('image_width')),
        A.CLAHE(p=1.),
        A.ToFloat(p=1.),
        ToTensorV2()
    ])

    return train_augs, val_augs


def get_optimizer(params, algorithm: str, args: dict):
    optimizer = None
    if algorithm == 'sgd':
        optimizer = partial(
            torch.optim.SGD,
            params=params,
            momentum=args.get('momentum'),
            weight_decay=args.get('weight_decay'),
            nesterov=args.get('nesterov'),
            eps=OPTIMIZER_EPS,
            # maximize=OPTIMIZER_MAXIMIZE,
            # foreach=OPTIMIZER_FOREACH
        )
    elif algorithm == 'adam':
        optimizer = partial(
            torch.optim.Adam,
            params=params,
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            amsgrad=args.get('amsgrad'),
            eps=OPTIMIZER_EPS,
            # maximize=OPTIMIZER_MAXIMIZE,
            # foreach=OPTIMIZER_FOREACH
        )
    elif algorithm == 'adamw':
        optimizer = partial(
            torch.optim.AdamW,
            params=params,
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            amsgrad=args.get('amsgrad'),
            eps=OPTIMIZER_EPS,
            # maximize=OPTIMIZER_MAXIMIZE,
            # foreach=OPTIMIZER_FOREACH
        )
    elif algorithm == 'sparse_adam':
        optimizer = partial(
            torch.optim.SparseAdam,
            params=params,
            betas=args.get('betas'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'nadam':
        optimizer = partial(
            torch.optim.NAdam,
            params=params,
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            momentum_decay=args.get('momentum_decay'),
            eps=OPTIMIZER_EPS,
            # foreach=OPTIMIZER_FOREACH
        )
    elif algorithm == 'adamax':
        optimizer = partial(
            torch.optim.Adamax,
            params=params,
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            eps=OPTIMIZER_EPS,
            # maximize=OPTIMIZER_MAXIMIZE,
            # foreach=OPTIMIZER_FOREACH
        )
    elif algorithm == 'adadelta':
        optimizer = partial(
            torch.optim.Adadelta,
            params=params,
            rho=args.get('rho'),
            weight_decay=args.get('weight_decay'),
            eps=OPTIMIZER_EPS,
            # maximize=OPTIMIZER_MAXIMIZE,
            # foreach=OPTIMIZER_FOREACH
        )
    elif algorithm == 'adagrad':
        optimizer = partial(
            torch.optim.Adadelta,
            params=params,
            lr_decay=args.get('lr_decay'),
            weight_decay=args.get('weight_decay'),
            eps=OPTIMIZER_EPS,
            # maximize=OPTIMIZER_MAXIMIZE,
            # foreach=OPTIMIZER_FOREACH
        )
    return optimizer(lr=args.get('lr'))


def get_device(gpu_id: int = 0, logger: logging.Logger = None):
    n_gpus = torch.cuda.device_count()

    print('Available GPUs:')
    print(f'\t- Number of GPUs: {n_gpus}')
    device = 'cpu'
    if n_gpus > 0:
        try:
            if -1 < gpu_id < n_gpus - 1:
                print(f'Setting GPU to: {gpu_id}')

                device = f'cuda:{gpu_id}'

                print(f'''
                ======================
                = Running on {device}  =
                ======================
                ''')
            elif gpu_id > n_gpus - 1:

                device = f'cuda'
                print(f'''
                =====================================
                = Running on all the available GPUs =
                =====================================
                    ''')

        except RuntimeError as err:
            if isinstance(logger, logging.Logger):
                logger.exception(err)

    return device


def to_pickle(file, name: str, save_dir: str or pathlib.Path):
    os.makedirs(save_dir, exist_ok=True)

    pkl.dump(file, (save_dir / (name + '.pkl')).open(mode='wb'))


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # - GENERAL PARAMETERS
    parser.add_argument('--gpu_id', type=int, choices=[gpu_id for gpu_id in range(-1, torch.cuda.device_count() - 1)], default=0 if torch.cuda.device_count() > 0 else -1, help='The ID of the GPU (if there is any) to run the network on (e.g., --gpu_id 1 will run the network on GPU #1 etc.)')

    parser.add_argument('--train_continue', default=False, action='store_true', help=f'If to continue the training from the checkpoint saved at \'{CHECKPOINT_DIR}\'')
    parser.add_argument('--train_data_dir', type=str, default=TRAIN_DIR, help='The path to the directory where the images and corresponding masks are stored')
    parser.add_argument('--inference_data_dir', type=str, default=INFERENCE_DIR, help='The path to the directory where the inference images and corresponding masks are stored')

    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='The path to the directory where the outputs will be placed')

    parser.add_argument('--image_width', type=int, default=IMAGE_WIDTH, help='The width of the images that will be used for network training and inference. If not specified, will be set to IMAGE_WIDTH as in general_configs.py file.')
    parser.add_argument('--image_height', type=int, default=IMAGE_HEIGHT, help='The height of the images that will be used for network training and inference. If not specified, will be set to IMAGE_HEIGHT as in general_configs.py file.')

    parser.add_argument('--in_channels', type=int, default=IN_CHANNELS, help='The number of channels in an input image (e.g., 3 for RGB, 1 for Grayscale etc)')
    parser.add_argument('--out_channels', type=int, default=OUT_CHANNELS, help='The number of channels in the output image (e.g., 3 for RGB, 1 for Grayscale etc)')

    # - TRAINING
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='The number of samples in each batch')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='The number of workers to load the data')
    parser.add_argument('--val_prop', type=float, default=VAL_PROP, help=f'The proportion of the data which will be set aside, and be used in the process of validation')
    parser.add_argument('--checkpoint_file', type=str, default=CHECKPOINT_FILE, help=f'The path to the file which contains the checkpoints of the model')

    # - OPTIMIZERS
    # optimizer
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adamw', 'sparse_adam', 'nadam', 'adadelta', 'adamax', 'adagrad'], default=OPTIMIZER,  help=f'The optimizer to use')

    parser.add_argument('--optimizer_lr', type=float, default=OPTIMIZER_LR, help=f'The initial learning rate of the optimizer')
    parser.add_argument('--optimizer_lr_decay', type=float, default=OPTIMIZER_LR_DECAY, help=f'The learning rate decay for Adagrad optimizer')
    parser.add_argument('--optimizer_beta_1', type=float, default=OPTIMIZER_BETA_1, help=f'The exponential decay rate for the 1st moment estimates (Adam, Nadam, Adamax)')
    parser.add_argument('--optimizer_beta_2', type=float, default=OPTIMIZER_BETA_2, help=f'The exponential decay rate for the 2st moment estimates (Adam, Nadam, Adamax)')
    parser.add_argument('--optimizer_rho', type=float, default=OPTIMIZER_RHO, help=f'The decay rate (Adadelta, RMSprop)')
    parser.add_argument('--optimizer_amsgrad', default=False, action='store_true', help=f'If to use the Amsgrad function (Adam, Nadam, Adamax)')

    parser.add_argument('--optimizer_weight_decay', type=float, default=OPTIMIZER_WEIGHT_DECAY, help=f'The weight decay for ADAM, NADAM')
    parser.add_argument('--optimizer_momentum', type=float, default=OPTIMIZER_MOMENTUM, help=f'The momentum for SGD')
    parser.add_argument('--optimizer_dampening', type=float, default=OPTIMIZER_DAMPENING, help=f'The dampening for momentum')
    parser.add_argument('--optimizer_momentum_decay', type=float, default=OPTIMIZER_MOMENTUM_DECAY, help=f'The momentum for NADAM')
    parser.add_argument('--optimizer_nesterov', default=False, action='store_true', help=f'If to use the Nesterov momentum (SGD)')
    parser.add_argument('--optimizer_centered', default=False, action='store_true', help=f'If True, gradients are normalized by the estimated variance of the gradient; if False, by the un-centered second moment. Setting this to True may help with training, but is slightly more expensive in terms of computation and memory. (RMSprop)')

    # - CALLBACKS
    parser.add_argument('--no_drop_block', default=False, action='store_true', help=f'If to use the drop_block in the network')
    parser.add_argument('--drop_block_keep_prob', type=float, help=f'The probability to keep the block')
    parser.add_argument('--drop_block_block_size', type=int, help=f'The size of the block to drop')

    parser.add_argument('--kernel_regularizer_type', type=str, choices=['l1', 'l2', 'l1l2'], default=KERNEL_REGULARIZER_TYPE, help=f'The type of the regularization')
    parser.add_argument('--kernel_regularizer_l1', type=float, default=KERNEL_REGULARIZER_L1, help=f'The strength of the L1 regularization')
    parser.add_argument('--kernel_regularizer_l2', type=float, default=KERNEL_REGULARIZER_L2, help=f'The strength of the L2 regularization')
    parser.add_argument('--kernel_regularizer_factor', type=float, default=KERNEL_REGULARIZER_FACTOR, help=f'The strength of the orthogonal regularization')
    parser.add_argument('--kernel_regularizer_mode', type=str, choices=['rows', 'columns'], default=KERNEL_REGULARIZER_MODE, help=f"The mode ('columns' or 'rows') of the orthogonal regularization")

    parser.add_argument('--wandb', default=False, action='store_true', help=f'If to use the Weights and Biases board')
    parser.add_argument('--load_model', default=False, action='store_true', help=f'If to load the model')

    return parser


def train_fn(data_loader, model, optimizer, loss_fn, scaler, device: str):
    # - TRAIN
    data_loop = tqdm(data_loader)

    print('\nTraining ...')
    losses = np.array([])
    true_seg_msrs = np.array([])
    pred_seg_msrs = np.array([])
    for btch_idx, (imgs, _, aug_segs, seg_msrs) in enumerate(data_loop):
        imgs = imgs.to(device=device)
        aug_segs = aug_segs.to(device=device)
        seg_msrs = seg_msrs.unsqueeze(1).to(device=device)

        # Forward pass
        with torch.cuda.amp.autocast():
            preds = model(imgs, aug_segs)
            loss = loss_fn(preds, seg_msrs)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loss = loss.item()
        true_seg_msr = seg_msrs.detach().cpu().numpy().mean()
        pred_seg_msr = preds.detach().cpu().numpy().mean()

        data_loop.set_postfix(loss=loss, true_vs_pred=f'{true_seg_msr:.3f}/{pred_seg_msr:.3f}({100 - (100 * pred_seg_msr / (true_seg_msr + EPSILON)):.2f}%)')

        losses = np.append(losses, loss)
        true_seg_msrs = np.append(true_seg_msrs, true_seg_msr)
        pred_seg_msrs = np.append(pred_seg_msrs, pred_seg_msr)

    return losses.mean(), true_seg_msrs.mean(), pred_seg_msrs.mean()


def val_fn(data_loader, model, loss_fn, device: str):
    # - VALIDATION
    print('\nValidating ...')
    model.eval()
    data_loop = tqdm(data_loader)
    losses = np.array([])
    true_seg_msrs = np.array([])
    pred_seg_msrs = np.array([])
    for btch_idx, (imgs, _, aug_masks, seg_msrs) in enumerate(data_loop):
        imgs = imgs.to(device=device)
        aug_masks = aug_masks.to(device=device)
        seg_msrs = seg_msrs.unsqueeze(1).to(device=device)

        with torch.no_grad():
            # Forward pass
            with torch.cuda.amp.autocast():
                preds = model(image=imgs, mask=aug_masks)
                loss = loss_fn(preds, seg_msrs).item()

        # Update tqdm loop
        true_seg_msr = seg_msrs.detach().cpu().numpy().mean()
        pred_seg_msr = preds.detach().cpu().numpy().mean()

        data_loop.set_postfix(loss=loss, true_vs_pred=f'{true_seg_msr:.3f}/{pred_seg_msr:.3f}({100 - (100 * pred_seg_msr / (true_seg_msr + EPSILON)):.2f}%)')

        losses = np.append(losses, loss)
        true_seg_msrs = np.append(true_seg_msrs, true_seg_msr)
        pred_seg_msrs = np.append(pred_seg_msrs, pred_seg_msr)

    model.train()

    return losses.mean(), true_seg_msrs.mean(), pred_seg_msrs.mean()


def test_fn(data_loader, model, loss_fn, device: str):
    # - VALIDATION
    print('\nTesting ...')
    model.eval()
    data_loop = tqdm(data_loader)
    losses = np.array([])
    true_seg_msrs = np.array([])
    pred_seg_msrs = np.array([])
    for btch_idx, (imgs, _, aug_masks, seg_msrs) in enumerate(data_loop):
        imgs = imgs.to(device=device)
        aug_masks = aug_masks.to(device=device)
        seg_msrs = seg_msrs.unsqueeze(1).to(device=device)

        with torch.no_grad():
            # Forward pass
            with torch.cuda.amp.autocast():
                preds = model(image=imgs, mask=aug_masks)
                loss = loss_fn(preds, seg_msrs).item()

        # Update tqdm loop
        true_seg_msr = seg_msrs.detach().cpu().numpy().mean()
        pred_seg_msr = preds.detach().cpu().numpy().mean()

        data_loop.set_postfix(loss=loss, true_vs_pred=f'{true_seg_msr:.3f}/{pred_seg_msr:.3f}({100 - (100 * pred_seg_msr / (true_seg_msr + EPSILON)):.2f}%)')

        losses = np.append(losses, loss)
        true_seg_msrs = np.append(true_seg_msrs, true_seg_msr)
        pred_seg_msrs = np.append(pred_seg_msrs, pred_seg_msr)

    model.train()

    return losses.mean(), true_seg_msrs.mean(), pred_seg_msrs.mean()


def train_model(data_dir, epochs, args, device: str, save_dir: pathlib.Path, logger: logging.Logger = None):
    # - Print some examples
    train_dir = save_dir / 'train'
    os.makedirs(save_dir, exist_ok=True)

    plots_dir = train_dir / 'plots'
    os.makedirs(plots_dir, exist_ok=True)

    val_preds_dir = train_dir / 'val_preds'
    os.makedirs(val_preds_dir, exist_ok=True)

    chkpt_dir = save_dir / 'checkpoints'
    os.makedirs(chkpt_dir, exist_ok=True)

    # - Build the model
    model_configs = get_model_configs(configs_file=MODEL_CONFIGS_FILE, logger=logger)
    model = RibCage(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        input_image_shape=(args.image_height, args.image_width),
        conv2d_out_channels=model_configs.get('conv2d_blocks')['out_channels'],
        conv2d_kernel_sizes=model_configs.get('conv2d_blocks')['kernel_sizes'],
        fc_out_features=model_configs.get('fc_blocks')['out_features'],
        output_dir=args.output_dir,
        logger=logger
    ).to(device)

    if args.load_model:
        chkpt_fl = pathlib.Path(args.checkpoint_file)
        if chkpt_fl.is_file():
            load_checkpoint(torch.load(chkpt_fl), model)

    # - Configure the optimizer
    optimizer = get_optimizer(
        params=model.parameters(),
        algorithm=args.optimizer,
        args=dict(
            lr=args.optimizer_lr,
            lr_decay=args.optimizer_lr_decay,
            betas=(args.optimizer_beta_1, args.optimizer_beta_2),
            weight_decay=args.optimizer_weight_decay,
            momentum=args.optimizer_momentum,
            momentum_decay=args.optimizer_momentum_decay,
            dampening=args.optimizer_dampening,
            rho=args.optimizer_rho,
            nesterov=args.optimizer_nesterov,
            amsgrad=args.optimizer_amsgrad
        )
    )

    # - Get the data loaders
    train_data_loader, val_data_loader, _ = get_data_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        val_prop=VAL_PROP,
        train_augs=train_augs,
        val_augs=val_augs,
        test_augs=None,
        logger=logger
    )

    # - Train loop
    best_loss = np.inf
    no_imprv_epchs = 0

    train_losses = np.array([])
    train_true = np.array([])
    train_pred = np.array([])

    val_losses = np.array([])
    val_true = np.array([])
    val_pred = np.array([])

    scaler = torch.cuda.amp.GradScaler()
    for epch in range(epochs):
        print(f'\n== Epoch: {epch + 1}/{args.epochs} ({100 * (epch + 1) / args.epochs:.2f}% done) ==')

        train_loss, train_true_mean, train_pred_mean = train_fn(data_loader=train_data_loader, model=model, optimizer=optimizer, loss_fn=LOSS, scaler=scaler, device=device)
        train_losses = np.append(train_losses, train_loss)
        train_true = np.append(train_true, train_true_mean)
        train_pred = np.append(train_pred, train_pred_mean)

        val_loss, val_true_mean, val_pred_mean = val_fn(data_loader=val_data_loader, model=model, loss_fn=LOSS, device=device)
        val_losses = np.append(val_losses, val_loss)
        val_true = np.append(val_true, val_true_mean)
        val_pred = np.append(val_pred, val_pred_mean)

        # - Save the best model
        if val_loss < best_loss:

            # - Save checkpoint
            print(f'<!!> val_loss improved from {best_loss:.3f} -> {val_loss:.3f}')

            # - Update the best loss
            best_loss = val_loss

            checkpoint = dict(
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            save_checkpoint(state=checkpoint, filename=str(chkpt_dir / f'best_val_loss_chkpt.pth.tar'))

            # - Reset the non-improvement counter
            no_imprv_epchs = 0

        else:
            print(f'<!> val_loss ({val_loss:.3f}) did not improved from the last best_loss value ({best_loss:.3f})')

            # - Increase the non-improvement counter
            no_imprv_epchs += 1

        # - Plot progress
        fig, ax = plt.subplots()

        # - Loss
        ax.plot(np.arange(epch + 1), train_losses, color='g', label='Train loss')
        ax.plot(np.arange(epch + 1), val_losses, color='r', label='Validation loss')

        # - Seg measure
        ax.plot(np.arange(epch + 1), train_true, color='g', marker='<', label='Train true seg measure')
        ax.plot(np.arange(epch + 1), train_pred, color='r', marker='<', label='Train predicted seg measure')
        ax.plot(np.arange(epch + 1), val_true, color='g', marker='o', label='Validation true seg measure')
        ax.plot(np.arange(epch + 1), val_pred, color='r', marker='o', label='Validation predicted seg measure')

        plt.legend()
        plt.savefig(f'{plots_dir}/train_stats.png')

        # - Print stats
        # -> loss
        train_loss = train_losses[-1]
        val_loss = val_losses[-1]
        # -> train seg measure
        train_pred_seg_msr = train_pred[-1]
        train_true_seg_msr = train_true[-1]
        # -> val seg measure
        val_true_seg_msr = val_true[-1]
        val_pred_seg_msr = val_pred[-1]
        print(f'''
        Epoch {epch + 1} Stats (train vs val):
            - Loss: 
                train - {train_loss:.4f}
                val - {val_loss:.4f}
            - Seg Measure: 
                train - {train_true_seg_msr:.3f} / {train_pred_seg_msr:.3f} ({100 - (train_pred_seg_msr.mean() * 100 / (train_true_seg_msr + EPSILON)):.2f}%)
                val - {val_true_seg_msr:.3f} / {val_pred_seg_msr:.3f} ({100 - (val_pred_seg_msr * 100 / (val_true_seg_msr + EPSILON)):.2f}%)
        ''')

        # > Early Stopping
        if EARLY_STOPPING:
            if no_imprv_epchs >= EARLY_STOPPING_PATIENCE:
                print(f'<x> No improvement was recorded for {EARLY_STOPPING_PATIENCE} epochs - stopping the training!')
                break

        # > LR Reduction on Plateau
        if REDUCE_LR_ON_PLATEAU:
            if no_imprv_epchs >= REDUCE_LR_ON_PLATEAU_PATIENCE:
                lr = optimizer.param_groups[0]['lr']
                new_lr = REDUCE_LR_ON_PLATEAU_FACTOR * lr
                if new_lr < REDUCE_LR_ON_PLATEAU_MIN:
                    print(f'<x> The lr ({new_lr:.3f}) was reduced beyond its smallest possible value ({REDUCE_LR_ON_PLATEAU_MIN:.3f}) - stopping the training!')
                    break

                optimizer.param_groups[0]['lr'] = new_lr

                print(f'<!> No improvement was recorded for {REDUCE_LR_ON_PLATEAU_PATIENCE} epochs - reducing lr by factor {REDUCE_LR_ON_PLATEAU_FACTOR:.3f}, from {lr:.3f} -> {new_lr:.3f}!')

    return model


def test_model(model, data_dir, args, device: str, seg_dir_postfix: str, image_prefix: str, seg_prefix: str, save_dir: pathlib.Path, logger: logging.Logger = None):
    _, _, test_data_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=1,
        train_augs=None,
        val_augs=None,
        seg_dir_postfix=seg_dir_postfix,
        image_prefix=image_prefix,
        seg_prefix=seg_prefix,
        val_prop=0.,
        test_augs=test_augs,
        logger=logger
    )

    loss, true_seg_msrs, pred_seg_msrs = test_fn(data_loader=test_data_loader, model=model, loss_fn=LOSS, device=device)

    print(f'''
    Test Results (on files from \'{data_dir}\' dir):
        Loss: 
            - {loss:.4f}
        Seg Measure: 
            - {true_seg_msrs:.3f} / {pred_seg_msrs:.3f} ({100 - (pred_seg_msrs * 100 / (true_seg_msrs + EPSILON)):.2f}%)
    ''')
