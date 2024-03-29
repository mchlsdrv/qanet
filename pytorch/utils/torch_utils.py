import logging
import os
import pathlib

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import augs
from utils.augs import inference_augs
from utils.aux_funcs import (
    data_not_found_err,
    check_file,
    get_model_configs, line_plot, scatter_plot
)
from configs.general_configs import (
    VAL_PROP,
    TR_LOSS,
    EPSILON,
    MIN_IMPROVEMENT_DELTA,
)
from .torch_aux import load_checkpoint
from .torch_train_utils import get_optimizer, save_checkpoint_on_improvement, lr_reduction_scheduler, early_stopping, reduce_lr_on_plateau
from .. custom.torch_models import RibCage

from . torch_data_utils import (
    get_data_loaders,
)

from configs.general_configs import (
    MODEL_CONFIGS_FILE,
)


def train_fn(data_loader, model, optimizer, loss_fn, scaler, device: str):

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
            # TODO: maybe use MAE instead
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

    return losses.mean(), true_seg_msrs, pred_seg_msrs, abs_err_mus.mean(), abs_err_stds.sum() / ((len(abs_err_stds) - 1) + EPSILON)


def val_fn(data_loader, model, loss_fn, device: str):
    # - VALIDATION
    print('\n> Validating ...')
    model.eval()
    data_loop = tqdm(data_loader)
    losses = np.array([])
    true_seg_msrs_history = np.array([])
    pred_seg_msrs_history = np.array([])
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
        true_seg_msrs_history = np.append(true_seg_msrs_history, true_seg_msrs)

        pred_seg_msrs = preds.detach().cpu().numpy()
        pred_seg_msrs_history = np.append(pred_seg_msrs_history, pred_seg_msrs)

        abs_errs = np.abs(true_seg_msrs - pred_seg_msrs)
        abs_err_mu, abs_err_std = abs_errs.mean(), abs_errs.std()

        losses = np.append(losses, loss)
        abs_err_mus = np.append(abs_err_mus, abs_err_mu)
        abs_err_stds = np.append(abs_err_stds, abs_err_std)

        data_loop.set_postfix(loss=loss, batch_err=f'{abs_err_mu:.3f}+/-{abs_err_std:.4f} ({100 * abs_err_std / (abs_err_mu + EPSILON):.3f}%)')

    model.train()

    return losses.mean(), true_seg_msrs_history, pred_seg_msrs_history, abs_err_mus.mean(), abs_err_stds.sum() / ((len(abs_err_stds) - 1) + EPSILON)


def train_model(data_file, epochs, args, device: str, output_dir: pathlib.Path, logger: logging.Logger = None):
    model = None
    if check_file(file_path=data_file):
        data = np.load(str(data_file), allow_pickle=True)

        # - Print some examples
        plots_dir = output_dir / 'plots'
        os.makedirs(plots_dir, exist_ok=True)

        chkpt_dir = output_dir / 'checkpoints'
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

        if args.continue_train:
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

            # - Train Model
            train_loss, train_true_seg_msrs, train_pred_seg_msrs, train_err_mu, train_err_std = train_fn(data_loader=train_data_loader, model=model, optimizer=optimizer, loss_fn=TR_LOSS, scaler=scaler, device=device)

            # -*- Add train history
            train_losses = np.append(train_losses, train_loss)
            train_err_mus = np.append(train_err_mus, train_err_mu)
            train_err_stds = np.append(train_err_stds, train_err_std)

            # - Validate Model
            val_loss, val_true_seg_msrs, val_pred_seg_msrs, val_err_mu, val_err_std = val_fn(data_loader=val_data_loader, model=model, loss_fn=TR_LOSS, device=device)

            # -*- Add val history
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
            line_plot(x=np.arange(epch + 1), ys=[train_losses, val_losses], suptitle='Train / Validation Loss Plot', labels=['Train', 'Validation'], colors=('r', 'g'), save_file=f'{plots_dir}/train_val_loss.png', logger=logger)

            # - Seg measure mean error
            line_plot(x=np.arange(epch + 1), ys=[train_err_mus, val_err_mus], suptitle='Train / Validation Error Plot', labels=['Train', 'Validation'], colors=('r', 'g'), save_file=f'{plots_dir}/train_val_error.png', logger=logger)

            # - Scatter plot
            # > Train
            scatter_plot(x=train_true_seg_msrs.flatten(), y=train_pred_seg_msrs.flatten(), save_file=f'{plots_dir}/scatter/train/epoch_{epch + 1}.png', logger=logger)

            # > Validation
            scatter_plot(x=val_true_seg_msrs.flatten(), y=val_pred_seg_msrs.flatten(), save_file=f'{plots_dir}/scatter/val/epoch_{epch + 1}.png', logger=logger)

            # - Print stats
            print(f'''
    Epoch {epch + 1} Stats (train vs val):
        - Loss: 
            train - {train_loss:.6f}
            val - {val_loss:.6f}
        - Seg Measure Error: 
            train - {train_err_mu:.4f} +/- {train_err_std:.5f} ({100 - (train_err_std * 100 / (train_err_mu + EPSILON)):.3f}%)
            val - {val_err_mu:.4f} +/- {val_err_std:.5f} ({100 - (val_err_std * 100 / (val_err_mu + EPSILON)):.3f}%)
            ''')

            # - CALLBACKS

            # > Early Stopping
            stop_training_es = early_stopping(no_improvement_epochs=no_imprv_epchs)

            # > LR Reduction on Plateau
            stop_training_rlp = reduce_lr_on_plateau(no_improvement_epochs=no_imprv_epchs, optimizer=optimizer)

            # > LR Reduction Scheduler
            stop_training_lrr = lr_reduction_scheduler(epoch=epch, optimizer=optimizer)

            if stop_training_es or stop_training_rlp or stop_training_lrr:
                break
    else:
        data_not_found_err(data_file=data_file, logger=logger)

    return model


def test_model(model, data_file: str or pathlib.Path, args, device: str, output_dir: pathlib.Path, logger: logging.Logger = None):
    if check_file(file_path=data_file):

        data = np.load(str(data_file), allow_pickle=True)

        # - Print some examples
        plots_dir = output_dir / 'plots'
        os.makedirs(plots_dir, exist_ok=True)

        _, _, test_data_loader = get_data_loaders(
            data=data,
            train_batch_size=0,
            train_augs=None,
            val_augs=None,
            val_prop=0.,
            test_augs=augs.test_augs,
            logger=logger
        )

        loss, true_seg_msrs, pred_seg_msrs, err_mu, err_std = val_fn(data_loader=test_data_loader, model=model, loss_fn=TR_LOSS, device=device)

        # - Plot progress

        # - Scatter plot
        scatter_plot(x=true_seg_msrs, y=pred_seg_msrs, save_file=f'{plots_dir}/test_scatter.png', logger=logger)

        # - Print stats
        print(f'''
    Test Results (on files from \'{data_file}\' dir):
        Loss: 
            - {loss:.6f}
        - Seg Measure Error: 
            test - {err_mu:.4f} +/- {err_std:.5f} ({100 - (err_mu * 100 / (err_mu + EPSILON)):.3f}%)
        ''')
    else:
        data_not_found_err(data_file=data_file, logger=logger)


def infer(data_files, model, device: str, output_dir: pathlib.Path):
    # - VALIDATION
    print('\n> Running Inference ...')
    model.eval()

    data_loop = tqdm(data_files)

    inf_mean_preds = np.array([])

    inf_augs = inference_augs()

    plots_dir = output_dir / 'plots'
    os.makedirs(plots_dir, exist_ok=True)

    for idx, (img_fl, mask_fl) in enumerate(data_loop):
        # - Create figure for plots
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

        # - Load the image and plot it
        img = cv2.imread(str(img_fl), -1)
        ax[0].imshow(img, cmap='gray')

        # - Load the corresponding mask and plot it
        mask = cv2.imread(str(mask_fl), -1)
        ax[1].imshow(mask, cmap='gray')

        # - Augment the image and the mask
        augs = inf_augs(image=img, mask=mask)
        img, mask = np.expand_dims(augs.get('image'), 0), np.expand_dims(augs.get('mask'), 0)

        # - Convert to tensors and move to device
        img = torch.tensor(np.expand_dims(img, 0), dtype=torch.float).to(device=device)
        mask = torch.tensor(np.expand_dims(mask, 0).astype(np.float32), dtype=torch.float).to(device=device)

        with torch.no_grad():
            # Forward pass
            with torch.cuda.amp.autocast():
                preds = model(image=img, mask=mask)

        preds = preds.cpu().numpy().flatten()
        preds_mu = preds.mean()
        inf_mean_preds = np.append(inf_mean_preds, preds_mu)

        # - Plot preds
        fig.suptitle(f'Estimated Seg Measure: {preds_mu:.4f}')
        plt.savefig(plots_dir / f'{idx}.png')
        plt.close(fig)

        # - Update the progress bar
        data_loop.set_postfix(prediction=f'{preds_mu:.4f}')

    model.train()

    print(f'''
    ======================================
    Final Preds:
        - {inf_mean_preds.mean():.4f}
    ======================================
    ''')
