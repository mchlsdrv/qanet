import matplotlib.pyplot as plt
import numpy as np


def save_image_mask(image, mask, savefile: str = 'testimg.png'):
    fig, ax = plt.subplots(1, 2, figsize=(10, 20))
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(mask, cmap='gray')
    fig.savefig(savefile)


def save_mask_overlay(gt_mask, pred_mask, figsize: tuple = (20, 20),
                      savefile: str = 'testing.png'):
    msk_ovrly = np.zeros((*gt_mask.shape[:3], 3))
    msk_ovrly[..., 0] = gt_mask
    msk_ovrly[..., 1] = pred_mask
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(msk_ovrly)
    fig.savefig(savefile)
