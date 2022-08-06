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
    OPTIMIZER_EPS, EPSILON, MIN_IMPROVEMENT_DELTA, LR_REDUCTION_SCHEDULER, LR_REDUCTION_SCHEDULER_PATIENCE, LR_REDUCTION_SCHEDULER_FACTOR, LR_REDUCTION_SCHEDULER_MIN
)
from .. custom.torch_models import RibCage

from . torch_data_utils import (
    get_data_loaders,
)

from configs.general_configs import (
    MODEL_CONFIGS_FILE,
)

# - Global Variables
stop_training = False
best_loss = np.inf


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
    ====================================
    =       Running on all GPUs        =
    ====================================
                            ''')
            elif gpu_id < 0:
                device = 'cpu'
                print(f'''
    =====================================
    = Running on all the available GPUs =
    =====================================
                        ''')

        except RuntimeError as err:
            if isinstance(logger, logging.Logger):
                logger.exception(err)

    return device


# - CALLBACKS
# > Save Checkpoint on Improvement
def save_checkpoint_on_improvement(model, optimizer, output_dir: pathlib.Path):
    chckpnt = dict(
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    save_checkpoint(state=chckpnt, filename=str(output_dir / f'best_val_loss_chkpt.pth.tar'))


# > Early Stopping
def early_stopping(no_improvement_epochs: int):
    if EARLY_STOPPING:
        if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
            print(f'<x> No improvement was recorded for {EARLY_STOPPING_PATIENCE} epochs - stopping the training!')
            STOP_TRAINING = True
        else:
            if no_improvement_epochs > 0:
                print(f'<!> {no_improvement_epochs}/{EARLY_STOPPING_PATIENCE} of epochs without improvement recorded (i.e, stopping the training when counter reaches {EARLY_STOPPING_PATIENCE} consecutive epochs with improvements < {MIN_IMPROVEMENT_DELTA})!')


# > LR Reduction on Plateau
def reduce_lr_on_plateau(no_improvement_epochs: int, optimizer):
    global stop_training

    if REDUCE_LR_ON_PLATEAU:
        if no_improvement_epochs >= REDUCE_LR_ON_PLATEAU_PATIENCE:
            lr = optimizer.param_groups[0]['lr']
            new_lr = REDUCE_LR_ON_PLATEAU_FACTOR * lr
            if new_lr < REDUCE_LR_ON_PLATEAU_MIN:
                print(f'<x> The lr ({new_lr}) was reduced beyond its smallest possible value ({REDUCE_LR_ON_PLATEAU_MIN}) - stopping the training!')
                stop_training = True

            optimizer.param_groups[0]['lr'] = new_lr

            print(f'<!> No improvement was recorded for {REDUCE_LR_ON_PLATEAU_PATIENCE} epochs - reducing lr by factor {REDUCE_LR_ON_PLATEAU_FACTOR}, from {lr} -> {new_lr}!')


def lr_reduction_scheduler(epoch: int, optimizer):
    global stop_training

    if LR_REDUCTION_SCHEDULER:
        if epoch in LR_REDUCTION_SCHEDULER_PATIENCE:
            lr = optimizer.param_groups[0]['lr']
            new_lr = LR_REDUCTION_SCHEDULER_FACTOR * lr
            if new_lr < LR_REDUCTION_SCHEDULER_MIN:
                print(f'<x> The lr ({new_lr}) was reduced beyond its smallest possible value ({LR_REDUCTION_SCHEDULER_MIN}) - stopping the training!')
                stop_training = True

            optimizer.param_groups[0]['lr'] = new_lr

            print(f'<!> Reducing learning rate after epoch {epoch} > {LR_REDUCTION_SCHEDULER_PATIENCE} epochs - reducing lr by factor {LR_REDUCTION_SCHEDULER_FACTOR}, from {lr} -> {new_lr}!')


def train_fn(data_loader, model, optimizer, loss_fn, scaler, device: str):
    global stop_training
    global best_loss

    # - TRAIN
    data_loop = tqdm(data_loader)

    print('\n> Training ...')
    losses = np.array([])
    abs_err_mus = np.array([])
    abs_err_stds = np.array([])
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
        true_seg_msrs = seg_msrs.detach().cpu().numpy()
        pred_seg_msrs = preds.detach().cpu().numpy()

        abs_errs = np.abs(true_seg_msrs - pred_seg_msrs)
        abs_err_mu, abs_err_std = abs_errs.mean(), abs_errs.std()

        losses = np.append(losses, loss)
        abs_err_mus = np.append(abs_err_mus, abs_err_mu)
        abs_err_stds = np.append(abs_err_stds, abs_err_std)

        data_loop.set_postfix(loss=loss, batch_err=f'{abs_err_mu:.3f}+/-{abs_err_std:.4f} ({100 * abs_err_std / (abs_err_mu + EPSILON):.3f}%)')

    return losses.mean(), abs_err_mus.mean(), abs_err_stds.sum() / ((len(abs_err_stds) - 1) + EPSILON)


def val_fn(data_loader, model, loss_fn, device: str):
    # - VALIDATION
    print('\n> Validating ...')
    model.eval()
    data_loop = tqdm(data_loader)
    losses = np.array([])
    abs_err_mus = np.array([])
    abs_err_stds = np.array([])
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
        true_seg_msrs = seg_msrs.detach().cpu().numpy()
        pred_seg_msrs = preds.detach().cpu().numpy()

        abs_errs = np.abs(true_seg_msrs - pred_seg_msrs)
        abs_err_mu, abs_err_std = abs_errs.mean(), abs_errs.std()

        losses = np.append(losses, loss)
        abs_err_mus = np.append(abs_err_mus, abs_err_mu)
        abs_err_stds = np.append(abs_err_stds, abs_err_std)

        data_loop.set_postfix(loss=loss, batch_err=f'{abs_err_mu:.3f}+/-{abs_err_std:.4f} ({100 * abs_err_std / (abs_err_mu + EPSILON):.3f}%)')

    model.train()

    return losses.mean(), abs_err_mus.mean(), abs_err_stds.sum() / ((len(abs_err_stds) - 1) + EPSILON)


def test_fn(data_loader, model, loss_fn, device: str):
    # - VALIDATION
    print('\nTesting ...')
    model.eval()
    data_loop = tqdm(data_loader)
    losses = np.array([])
    abs_err_mus = np.array([])
    abs_err_stds = np.array([])
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
        true_seg_msrs = seg_msrs.detach().cpu().numpy()
        pred_seg_msrs = preds.detach().cpu().numpy()

        abs_errs = np.abs(true_seg_msrs - pred_seg_msrs)
        abs_err_mu, abs_err_std = abs_errs.mean(), abs_errs.std()

        losses = np.append(losses, loss)
        abs_err_mus = np.append(abs_err_mus, abs_err_mu)
        abs_err_stds = np.append(abs_err_stds, abs_err_std)

        data_loop.set_postfix(loss=loss, batch_err=f'{abs_err_mu:.3f} +/- {abs_err_std:.4f}({100 - (100 * abs_err_std / (abs_err_mu + EPSILON)):.3f}%)')

    model.train()

    return losses.mean(), abs_err_mus.mean(), abs_err_stds.sum() / ((len(abs_err_stds) - 1) + EPSILON)


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
        train_err_mus = np.array([])
        train_err_stds = np.array([])

        val_losses = np.array([])
        val_err_mus = np.array([])
        val_err_stds = np.array([])

        scaler = torch.cuda.amp.GradScaler()
        for epch in range(epochs):
            print(f'\n== Epoch: {epch + 1}/{args.epochs} ({100 * (epch + 1) / args.epochs:.2f}% done) ==')

            train_loss, train_err_mu, train_err_std = train_fn(data_loader=train_data_loader, model=model, optimizer=optimizer, loss_fn=TORCH_LOSS, scaler=scaler, device=device)

            # - Add train history
            train_losses = np.append(train_losses, train_loss)
            train_err_mus = np.append(train_err_mus, train_err_mu)
            train_err_stds = np.append(train_err_stds, train_err_std)

            val_loss, val_err_mu, val_err_std = val_fn(data_loader=val_data_loader, model=model, loss_fn=TORCH_LOSS, device=device)

            # - Add val history
            val_losses = np.append(val_losses, val_loss)
            val_err_mus = np.append(val_err_mus, val_err_mu)
            val_err_stds = np.append(val_err_stds, val_err_std)

            # - Save the best model
            if val_loss < best_loss - MIN_IMPROVEMENT_DELTA:

                print(f'<!!> val_loss improved by delta > {MIN_IMPROVEMENT_DELTA}, from {best_loss} to {val_loss}!')

                # - Save checkpoint
                save_checkpoint_on_improvement(model=model, optimizer=optimizer, output_dir=chkpt_dir)

                # - Update the best_loss
                best_loss = val_loss

                # - Reset the non-improvement counter
                no_imprv_epchs = 0

            else:
                print(f'<!> val_loss ({val_loss:.6f}) did not improved from the last best_loss value ({best_loss:.6f})')

                # - Increase the non-improvement counter
                no_imprv_epchs += 1

            # - Plot progress
            # - Loss
            fig, ax = plt.subplots()
            ax.plot(np.arange(epch + 1), train_losses, color='g', label='Train loss')
            ax.plot(np.arange(epch + 1), val_losses, color='r', label='Val loss')

            plt.legend()
            plt.savefig(f'{plots_dir}/train_val_loss.png')
            plt.close(fig)

            # - Seg measure mean error
            fig, ax = plt.subplots()
            ax.plot(np.arange(epch + 1), train_err_mus, color='g', label='Train E[error]')
            ax.plot(np.arange(epch + 1), val_err_mus, color='r', label='Val E[error]')

            plt.legend()
            plt.savefig(f'{plots_dir}/train_val_err_mu_std.png')
            plt.close(fig)

            # - Print stats
            # -> loss
            train_loss = train_losses[-1]
            val_loss = val_losses[-1]

            # -> train seg measure
            train_last_err_mu = train_err_mus[-1]
            train_last_err_std = train_err_stds[-1]

            # -> val seg measure
            val_last_err_mu = val_err_mus[-1]
            val_last_err_std = val_err_stds[-1]

            print(f'''
    Epoch {epch + 1} Stats (train vs val):
        - Loss: 
            train - {train_loss:.6f}
            val - {val_loss:.6f}
        - Seg Measure Error: 
            train - {train_err_mu:.4f} +/- {train_last_err_std:.5f} ({100 - (train_last_err_std * 100 / (train_last_err_mu + EPSILON)):.3f}%)
            val - {val_err_mu:.4f} +/- {val_last_err_std:.5f} ({100 - (val_last_err_std * 100 / (val_last_err_mu + EPSILON)):.3f}%)
            ''')

            # - CALLBACKS

            # > Early Stopping
            early_stopping(no_improvement_epochs=no_imprv_epchs)

            # > LR Reduction on Plateau
            reduce_lr_on_plateau(no_improvement_epochs=no_imprv_epchs, optimizer=optimizer)

            # > LR Reduction Scheduler
            lr_reduction_scheduler(epoch=epch, optimizer=optimizer)

            if stop_training:
                break
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
            - {loss:.6f}
        Seg Measure: 
            - {true_seg_msrs:.4f} / {pred_seg_msrs:.4f} ({100 - (pred_seg_msrs * 100 / (true_seg_msrs + EPSILON)):.2f}%)
        ''')
    else:
        data_not_found_err(data_file=data_file, logger=logger)
