import os
import logging
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import cv2


def plot_knn(knn: pd.DataFrame, save_dir: pathlib.Path, logger: logging.Logger = None):
    k = len(knn.loc[0, 'neighbors'][0])
    n_files = knn.shape[0]
    plots_dir = save_dir / 'plots'
    os.makedirs(plots_dir, exist_ok=True)

    for idx, file in enumerate(knn.loc[:, 'file']):
        fig, axs = plt.subplots(1, k, figsize=(100, 15), facecolor='#c0d6e4');
        file_name = file.split('/')[-1]
        file_name = file_name[:file_name.index('.')]
        distances = knn.loc[idx, 'distances'][0]
        neighbors = knn.loc[idx, 'neighbors'][0]
        for idx, (distance, neighbor) in enumerate(zip(distances, neighbors)):
            axs[idx].imshow(cv2.imread(neighbor))
            neighbor_name = neighbor.split('/')[-1]
            if not idx:
                axs[idx].set(title=f'{neighbor_name} (Original)')
            else:
                axs[idx].set(title=f'{neighbor_name} (Distance = {distance:.1f})')
            axs[idx].title.set_size(70)
        if isinstance(logger, logging.Logger):
            logger.info(f'Saving KNN image - {file_name} ({100 * idx / n_files:.1f}% done)')
        fig.savefig(plots_dir / (file_name + '.png'))

        plt.close(fig)

