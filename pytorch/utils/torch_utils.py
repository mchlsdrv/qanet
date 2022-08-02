import logging.config
from functools import partial

import logging
import os
import pathlib

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import augs
from utils.aux_funcs import data_not_found_err, check_file, get_model_configs
from configs.general_configs import (
    VAL_PROP,
    TORCH_LOSS,
    REDUCE_LR_ON_PLATEAU_MIN,
    REDUCE_LR_ON_PLATEAU,
    EARLY_STOPPING,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_ON_PLATEAU_FACTOR,
    REDUCE_LR_ON_PLATEAU_PATIENCE,
    OPTIMIZER_EPS, EPSILON
)
from .. custom.torch_models import RibCage

from . torch_data_utils import (
    get_data_loaders,
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
        )
    elif algorithm == 'adam':
        optimizer = partial(
            torch.optim.Adam,
            params=params,
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            amsgrad=args.get('amsgrad'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adamw':
        optimizer = partial(
            torch.optim.AdamW,
            params=params,
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            amsgrad=args.get('amsgrad'),
            eps=OPTIMIZER_EPS,
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
        )
    elif algorithm == 'adamax':
        optimizer = partial(
            torch.optim.Adamax,
            params=params,
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adadelta':
        optimizer = partial(
            torch.optim.Adadelta,
            params=params,
            rho=args.get('rho'),
            weight_decay=args.get('weight_decay'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adagrad':
        optimizer = partial(
            torch.optim.Adadelta,
            params=params,
            lr_decay=args.get('lr_decay'),
            weight_decay=args.get('weight_decay'),
            eps=OPTIMIZER_EPS,
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


def train_fn(data_loader, model, optimizer, loss_fn, scaler, device: str):
    # - TRAIN
    data_loop = tqdm(data_loader)

    print('\nTraining ...')
    losses = np.array([])
    true_seg_msrs = np.array([])
    pred_seg_msrs = np.array([])
    for btch_idx, (imgs, aug_segs, seg_msrs) in enumerate(data_loop):
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
    for btch_idx, (imgs, aug_masks, seg_msrs) in enumerate(data_loop):
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


def train_model(data_file, epochs, args, device: str, save_dir: pathlib.Path, logger: logging.Logger = None):
    if check_file(file_path=data_file):
        data = np.load(str(data_file), allow_pickle=True)

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
            data=data,
            train_batch_size=args.batch_size,
            val_prop=VAL_PROP,
            train_augs=augs.train_augs,
            val_augs=augs.val_augs,
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

            train_loss, train_true_mean, train_pred_mean = train_fn(data_loader=train_data_loader, model=model, optimizer=optimizer, loss_fn=TORCH_LOSS, scaler=scaler, device=device)
            train_losses = np.append(train_losses, train_loss)
            train_true = np.append(train_true, train_true_mean)
            train_pred = np.append(train_pred, train_pred_mean)

            val_loss, val_true_mean, val_pred_mean = val_fn(data_loader=val_data_loader, model=model, loss_fn=TORCH_LOSS, device=device)
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

            # - CALLBACKS
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
    else:
        data_not_found_err(data_file=data_file, logger=logger)

    return model


def test_model(model, data_file: str or pathlib.Path, args, device: str, save_dir: pathlib.Path, logger: logging.Logger = None):
    if check_file(file_path=data_file):

        data = np.load(str(data_file), allow_pickle=True)

        _, _, test_data_loader = get_data_loaders(
            data=data,
            train_batch_size=0,
            train_augs=None,
            val_augs=None,
            val_prop=0.,
            test_augs=augs.test_augs,
            logger=logger
        )

        loss, true_seg_msrs, pred_seg_msrs = test_fn(data_loader=test_data_loader, model=model, loss_fn=TORCH_LOSS, device=device)

        print(f'''
        Test Results (on files from \'{data_file}\' dir):
            Loss: 
                - {loss:.4f}
            Seg Measure: 
                - {true_seg_msrs:.3f} / {pred_seg_msrs:.3f} ({100 - (pred_seg_msrs * 100 / (true_seg_msrs + EPSILON)):.2f}%)
        ''')
    else:
        data_not_found_err(data_file=data_file, logger=logger)
