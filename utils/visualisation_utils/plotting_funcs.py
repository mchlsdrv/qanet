import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def plot_scatter(x: np.ndarray, y: np.ndarray, figsize: tuple = (20, 10), save_file: pathlib.Path = None):
    fig, ax = plt.subplots(figsize=figsize)

    plt.rc('font', size=28)
    # ax.xaxis.label.set_fontsize(18)
    ax.scatter(x, y, label='True vs Predicted')
    ax.plot([0., 1.], [0., 1.], label='Perfect Match')
    ax.set(
        title='Seg-Measure - Ground Truth vs Predicted',

        xlabel='Ground Truth',
        xlim=[0., 1.],
        xticks=np.arange(0., 1.1, 0.1),

        ylabel='Predicted',
        ylim=[0., 1.],
        yticks=np.arange(0., 1.1, 0.1)
    )

    plt.legend()

    save_figure(figure=fig, save_file=save_file)

    return fig


def plot(images, labels, figsize=(25, 10), save_file: pathlib.Path = None) -> None:
    fig, ax = plt.subplots(1, len(images), figsize=figsize)
    for idx, (img, lbl) in enumerate(zip(images, labels)):
        ax[idx].imshow(img, cmap='gray')
        ax[idx].set_title(lbl)

    save_figure(figure=fig, save_file=save_file)


def save_figure(figure, save_file):
    if isinstance(save_file, pathlib.Path):
        os.makedirs(save_file.parent, exist_ok=True)
        figure.savefig(str(save_file))
        plt.close(figure)


def plot_seg_measure_histogram(seg_measures: np.ndarray, bin_width: float = .1, figsize: tuple = (25, 10), density: bool = False, save_file: pathlib.Path = None):
    vals, bins = np.histogram(seg_measures, bins=np.arange(0., 1. + bin_width, bin_width))
    if density:
        vals = vals / vals.sum()
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(bins[:-1], vals, width=bin_width, align='edge')
    ax.set(xlim=(0, 1), xticks=np.arange(0., 1.1, .1), xlabel='E[SM] (Mean Seg Measure)', ylabel='P(E[SM])')

    save_figure(figure=fig, save_file=save_file)


def plot_image_histogram(images: np.ndarray, labels: list, n_bins: int = 256, figsize: tuple = (25, 50), density: bool = False, save_file: pathlib.Path = None):
    fig, ax = plt.subplots(2, len(images), figsize=figsize)
    for idx, (img, lbl) in enumerate(zip(images, labels)):

        vals, bins = np.histogram(img, n_bins, density=True)
        if density:
            vals = vals / vals.sum()
        vals, bins = vals[1:], bins[1:][:-1]  # don't include the 0

        # - If there is only a single plot - no second dimension will be available, and it will result in an error
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
        hist_ax.set(xlim=(0, max_val), ylim=(0., 1.), yticks=np.arange(0., 1.1, .1), xlabel='I (Intensity)', ylabel='P(I)')

        # - Show the image
        img_ax.imshow(img, cmap='gray')
        img_ax.set_title(lbl)

    save_figure(figure=fig, save_file=save_file)
