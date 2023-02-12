import os
import io
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

import warnings

from utils.aux_funcs import categorical_2_rgb, err_log, check_pathable, str_2_path, info_log, print_pretty_message

mpl.use('Agg')  # <= avoiding the "Tcl_AsyncDelete: async handler deleted by
# the wrong thread" exception
plt.style.use('seaborn')  # <= using the seaborn plot style

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

sns.set()
RC = {
    'font.size': 32,
    'axes.labelsize': 50,
    'legend.fontsize': 30.0,
    'axes.titlesize': 32,
    'xtick.labelsize': 40,
    'ytick.labelsize': 40
}
sns.set_context(rc=RC)


def get_rgb_mask_figure(mask: np.ndarray, suptitle: str = '', title: str = '', figsize: tuple = (20, 20)):
    msk = categorical_2_rgb(mask)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(msk)
    ax.set(title=title)

    fig.suptitle(suptitle)

    return fig


def show_images(images, labels, suptitle='', figsize=(25, 10), save_file: pathlib.Path or str = None,
                verbose: bool = False, logger: logging.Logger = None) -> None:
    fig, ax = plt.subplots(1, len(images), figsize=figsize)
    for idx, (img, lbl) in enumerate(zip(images, labels)):
        ax[idx].imshow(img, cmap='gray')
        ax[idx].set_title(lbl)

    fig.suptitle(suptitle)

    save_figure(figure=fig, save_file=pathlib.Path(save_file),
                close_figure=True, verbose=verbose, logger=logger)


def line_plot(x: list or np.ndarray, ys: list or np.ndarray, suptitle: str,
              labels: list, colors: tuple = ('r', 'g', 'b'),
              save_file: pathlib.Path or str = None,
              logger: logging.Logger = None):
    fig, ax = plt.subplots()
    for y, lbl, clr in zip(ys, labels, colors):
        ax.plot(x, y, color=clr, label=lbl)

    plt.legend()

    try:
        save_figure(figure=fig, save_file=save_file, close_figure=False,
                    logger=logger)
    except Exception as err:
        err_log(logger=logger, message=f'{err}')


def get_hit_rate_plot_figure(true: np.ndarray, pred: np.ndarray,
                             hit_rate_percent: int = None,
                             figsize: tuple = (15, 15),
                             logger: logging.Logger = None):
    # - Calculate the absolute error of true vs pred
    abs_err = np.abs(true - pred)

    # - Create a histogram of the absolute errors
    abs_err_hist, abs_err_tolerance = np.histogram(abs_err, bins=100,
                                                   range=(0., 1.))

    # - Normalize the histogram
    abs_err_prob = abs_err_hist / np.sum(abs_err_hist)

    # - Plot the histogram
    fig, ax = plt.subplots(figsize=figsize)

    # > Calculate the cumulative probability of the density function of the
    # absolute errors
    abs_err_cum_sum_prob = np.cumsum(abs_err_prob)
    ax.plot(abs_err_tolerance[:-1], abs_err_cum_sum_prob, linewidth=2)
    ax.set(xlabel='Absolute Error Tolerance', xlim=(0, 1),
           xticks=np.arange(0.0, 1.2, 0.2), ylabel='Hit Rate', ylim=(0, 1),
           yticks=np.arange(0.2, 1.2, 0.2))

    # - Add a line representing the hit rate percentage with corresponding AET
    # value
    if isinstance(hit_rate_percent, int):
        # > Find the index of the hit_rate_percent
        abs_err_cum_sum_pct_idx = np.argwhere(
            abs_err_cum_sum_prob >= hit_rate_percent / 100).flatten().min()

        # > Find the real value of the hit_rate_percent
        cum_sum_err_pct = abs_err_cum_sum_prob[abs_err_cum_sum_pct_idx]

        # > Find the corresponding Absolute Error Tolerance to the
        # hit_rate_percent value
        abs_err_tolerance_pct = abs_err_tolerance[abs_err_cum_sum_pct_idx]

        # > Plot the horizontal line for the hit rate percentage
        ax.axhline(cum_sum_err_pct, xmax=abs_err_tolerance_pct)

        # > Plot the vertical line for the corresponding AET value
        ax.axvline(abs_err_tolerance_pct, ymax=cum_sum_err_pct)

        # > Add the corresponding AET value
        ax.text(x=abs_err_tolerance_pct, y=cum_sum_err_pct,
                s=f'AET={abs_err_tolerance_pct:.3f}')

    return fig, abs_err_hist, abs_err


def get_simple_scatter_plot_figure(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str,
                                   xticks: tuple or list = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                                   xlim: tuple or list = (0.0, 1.0),
                                   yticks: tuple or list = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                                   ylim: tuple or list = (0.0, 1.0),
                                   figsize: tuple = (20, 15),
                                   save_file: pathlib.Path or str = None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        x=x,
        y=y,
        alpha=.3,
        s=150,
    )
    ax.set(
        xlim=xlim,
        xlabel=xlabel,
        xticks=xticks,
        ylim=ylim,
        ylabel=ylabel,
        yticks=yticks
    )

    if check_pathable(path=save_file):
        save_file = str_2_path(path=save_file)
        fig.savefig(save_file)

    return fig, ax


def get_scatter_plot_figure(x: np.ndarray, y: np.ndarray, plot_type: str, logger: logging.Logger = None):
    D = pd.DataFrame({'GT Quality Value': x, 'Estimated Quality Value': y})
    g = sns.jointplot(
        x='GT Quality Value',
        y='Estimated Quality Value',
        marker='o',
        joint_kws={
            'scatter_kws': {
                'alpha': 0.3,
                's': 150
            }
        },
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        data=D,
        height=15,
        space=0.02,
        kind='reg'
    )

    # - Calculate pearson correlation
    rho, p = pearsonr(x, y)

    # - Calculate mean squared error
    mse = np.mean(np.square(x[:10] - y[:10]))

    g.ax_joint.annotate(
        f'$\\rho = {rho:.3f}, MSE = {mse:.3f}$',
        xy=(0.1, 0.9),
        xycoords='axes fraction',
        ha='left',
        va='center',
        bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'}
    )

    return g.figure, rho, p, mse


def save_figure(figure, save_file: pathlib.Path or str, overwrite: bool = False, close_figure: bool = False,
                verbose: bool = False, logger: logging.Logger = None):
    # - Convert save_file to path
    save_file = str_2_path(path=save_file)

    if isinstance(save_file, pathlib.Path):
        # - If the file does not exist or can be overwritten
        if not save_file.is_file() or overwrite:

            # - Create sub-path of the save file
            os.makedirs(save_file.parent, exist_ok=True)

            figure.savefig(str(save_file))
            if close_figure:
                plt.close(figure)
            if verbose:
                info_log(logger=logger,
                         message=f'Figure was saved to \'{save_file}\'')
        elif verbose:
            info_log(logger=logger,
                     message=f'Can not save figure - file \'{save_file}\' '
                             f'already exists and overwrite = {overwrite}!')
    elif verbose:
        info_log(logger=logger,
                 message=f'Can not save figure - save_file argument must be '
                         f'of type pathlib.Path or str, but {type(save_file)} '
                         f'was provided!')


def plot_seg_measure_histogram(seg_measures: np.ndarray, bin_width: float = .1, figsize: tuple = (25, 10),
                               density: bool = False, save_file: pathlib.Path = None):
    vals, bins = np.histogram(seg_measures,
                              bins=np.arange(0., 1. + bin_width, bin_width))
    if density:
        vals = vals / vals.sum()
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(bins[:-1], vals, width=bin_width, align='edge')
    ax.set(xlim=(0, 1), xticks=np.arange(0., 1.1, .1),
           xlabel='E[SM] (Mean Seg Measure)', ylabel='P(E[SM])')

    save_figure(figure=fig, save_file=save_file)


def plot_image_histogram(images: np.ndarray, labels: list, n_bins: int = 256, figsize: tuple = (25, 50),
                         density: bool = False, save_file: pathlib.Path = None):
    fig, ax = plt.subplots(2, len(images), figsize=figsize)
    for idx, (img, lbl) in enumerate(zip(images, labels)):

        vals, bins = np.histogram(img, n_bins, density=True)
        if density:
            vals = vals / vals.sum()
        vals, bins = vals[1:], bins[1:][:-1]  # don't include the 0

        # - If there is only a single plot - no second dimension will be
        # available, and it will result in an error
        if len(images) > 1:
            hist_ax = ax[0, idx]
            img_ax = ax[1, idx]
        else:
            hist_ax = ax[0]
            img_ax = ax[1]

        # - Plot the histogram
        hist_ax.bar(bins, vals)
        hist_ax.set_title('Intensity Histogram')
        max_val = 255 if img.max() > 1 else 1
        hist_ax.set(xlim=(0, max_val), ylim=(0., 1.),
                    yticks=np.arange(0., 1.1, .1), xlabel='I (Intensity)',
                    ylabel='P(I)')

        # - Show the image
        img_ax.imshow(img, cmap='gray')
        img_ax.set_title(lbl)

    save_figure(figure=fig, save_file=save_file)


def plot_mask_error(image: np.ndarray, mask: np.ndarray,
                    pred_mask: np.ndarray = None, suptitle: str = '',
                    title: str = '', figsize: tuple = (20, 20),
                    tensorboard_params: dict = None,
                    save_file: pathlib.Path = None, overwrite: bool = False):
    # - Prepare the mask overlap image
    msk_shp = mask.shape
    msk_dims = len(msk_shp)
    if msk_dims > 2:
        msk_shp = msk_shp[:-1]
    msk = np.zeros((*msk_shp, 3))
    msk[..., 1] = mask[..., 0] if msk_dims > 2 else mask

    # - If there is a predicted segmentation
    if isinstance(pred_mask, np.ndarray):
        msk[..., 0] = \
            pred_mask[..., 0] if len(pred_mask.shape) > 2 else pred_mask

    # - Convert instance segmentation to binary
    msk[msk > 0] = 1.

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap='gray')
    ax.imshow(msk, alpha=0.3)
    ax.set(title=title)

    fig.suptitle(suptitle)

    save_figure(figure=fig, save_file=save_file)

    if isinstance(tensorboard_params, dict):
        write_figure_to_tensorboard(
            writer=tensorboard_params.get('writer'),
            figure=fig,
            tag=tensorboard_params.get('tag'),
            step=tensorboard_params.get('step')
        )

    plt.close(fig)


def monitor_seg_error(gt_masks: np.ndarray, pred_masks: np.ndarray,
                      seg_measures: np.ndarray, images: np.ndarray = None,
                      n_samples: int = 5, figsize: tuple = (20, 10),
                      save_dir: str or pathlib.Path = './seg_errors'):
    save_dir = pathlib.Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    data = list(zip(gt_masks, pred_masks, seg_measures))

    for idx, (gt, pred, seg_msr) in zip(np.arange(n_samples), data):
        seg = np.zeros((*gt.shape[:-1], 3))
        seg[..., 0] = gt[..., 0]
        seg[..., 2] = pred[..., 0]
        seg[seg > 0] = 1.

        if isinstance(images, np.ndarray):
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            ax[0].imshow(images[idx], cmap='gray')
            ax[0].set(title='Original Image')
            ax[1].imshow(seg)
            ax[1].set(title=f'Seg Measure = {seg_msr:.4f}')
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(seg)
            ax.set(title=f'Seg Measure = {seg_msr:.4f}')

        fig.suptitle(f'GT (red) vs Pred (blue) ')

        plt.savefig(save_dir / f'item_{idx}.png')
        plt.close()


def plot_hist(data: np.ndarray or list, bins: np.ndarray,
              save_file: pathlib.Path = None, overwrite: bool = False):
    # - Plot histogram
    ds = pd.DataFrame(dict(heights=data))

    rc = {
        'font.size': 12,
        'axes.labelsize': 20,
        'legend.fontsize': 20.,
        'axes.titlesize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20
    }
    sns.set_context(rc=rc)
    dist_plot = sns.displot(ds['heights'], bins=bins, rug=True, kde=True)

    if isinstance(save_file, pathlib.Path) \
            and (not save_file.is_file() or overwrite):
        os.makedirs(save_file.parent, exist_ok=True)
        dist_plot.savefig(save_file)
    else:
        print(f'WARNING: could not save plot to \'{save_file}\' as it already '
              f'exists!')

    plt.close(dist_plot.figure)
    sns.set_context(rc=RC)

    print_pretty_message(message=f'An histogram was saved to: {save_file}', delimiter_symbol='*')


def get_image_from_figure(figure):
    buffer = io.BytesIO()

    plt.savefig(buffer, format='png')

    # plt.close(figure)
    buffer.seek(0)

    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image


def write_figure_to_tensorboard(writer, figure, tag: str, step: int):
    with tf.device('/cpu:0'):
        with writer.as_default():
            # -> Write the scatter plot
            tf.summary.image(
                tag,
                get_image_from_figure(figure=figure),
                step=step
            )
