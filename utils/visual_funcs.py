import os
import pathlib
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from utils.aux_funcs import (
    categorical_2_rgb,
    check_pathable,
    str_2_path,
    info_log,
    print_pretty_message, get_metrics
)

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


def get_image_figure(image: np.ndarray, suptitle: str = '', title: str = '', figsize: tuple = (20, 20)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap='gray')
    ax.set(title=title)

    fig.suptitle(suptitle)

    return fig


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

    # - Plot the perfect match line
    ax.plot((0, 0), (1, 1), 'g--', linewidth=1)

    if check_pathable(path=save_file):
        save_file = str_2_path(path=save_file)
        fig.savefig(save_file)

    return fig, ax


def get_scatter_plot_figure(x: np.ndarray, y: np.ndarray, plot_type: str = 'scatter', logger: logging.Logger = None):
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

    rho, p, mse = get_metrics(x=x, y=y)

    g.ax_joint.annotate(
        f'$\\rho = {rho:.3f} (p = {p:.4f}), MSE = {mse:.3f}$',
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


def plot_hist(data: np.ndarray or list, bins: np.ndarray, save_file: pathlib.Path = None, overwrite: bool = False):
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
