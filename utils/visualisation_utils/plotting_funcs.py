import os
import logging
import pathlib
import matplotlib.pyplot as plt
import cv2


def plot(images, labels, save_file: pathlib.Path = None) -> None:
    fig, ax = plt.subplots(1, len(images), figsize=(25, 10))
    for idx, (img, lbl) in enumerate(zip(images, labels)):
        ax[idx].imshow(img, cmap='gray')
        ax[idx].set_title(lbl)

    if isinstance(save_file, pathlib.Path):
        os.makedirs(save_file.parent, exist_ok=True)
        fig.savefig(str(save_file))
        plt.close(fig)
