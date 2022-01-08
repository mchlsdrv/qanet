import os
import pathlib
import numpy as np
import shutil
import logging
from sklearn.neighbors import NearestNeighbors
from utils.image_utils import image_funcs
from utils.visualisation_utils import plotting_funcs


def get_knn_files(X, files, k, algorithm='auto'):
    # Detect the k nearest neighbors
    nbrs_pred = NearestNeighbors(n_neighbors=k, algorithm=algorithm).fit(X)

    nbrs_distances = list()
    nbrs_files = list()
    for idx, (file, x) in enumerate(zip(files, X)):
        distances, nbrs_idxs = nbrs_pred.kneighbors(np.expand_dims(x, axis=0))

        nbrs_distances.append(distances)
        nbrs_files.append(files[nbrs_idxs])

    return nbrs_distances, nbrs_files


def get_priors_knn_df(model, preprocessing_func, k: int, train_data_dir: pathlib.Path, patch_height: int, patch_width: int, patch_optimization: bool, save_dir: pathlib.Path, knn_algorithm: str = 'auto'):
    priors_knn_df = None
    if patch_optimization:
        priors_knn_df = image_funcs.get_patch_transforms(images_root_dir=train_data_dir, model=model, preprocessing_func=preprocessing_func, patch_height=patch_height, patch_width=patch_width)
        X = np.array([x for x in priors_knn_df.loc[:, 'patch_transform'].values])
    else:
        priors_knn_df = image_funcs.get_mean_image_transforms(images_root_dir=train_data_dir, model=model, preprocessing_func=preprocessing_func, patch_height=patch_height, patch_width=patch_width)
        X = np.array([x for x in priors_knn_df.loc[:, 'image_mean_transform'].values])
    files = priors_knn_df.loc[:, 'file'].values

    nbrs_distances, nbrs_files = get_knn_files(X=X, files=files, k=k, algorithm=knn_algorithm)

    priors_knn_df.loc[:, 'distances'] = nbrs_distances
    priors_knn_df.loc[:, 'neighbors'] = nbrs_files

    os.makedirs(save_dir, exist_ok=True)
    priors_knn_df.to_pickle(save_dir / f'priors_knn_df.pkl')
    plotting_funcs.plot_knn(knn=priors_knn_df, save_dir=save_dir)

    return priors_knn_df


def classify(model, preprocessing_func, images_root_dir: pathlib.Path, patch_height: int, patch_width: int, output_dir: pathlib.Path, logger: logging.Logger = None):
    cls_df = image_funcs.get_mean_image_transforms(images_root_dir=images_root_dir, model=model, preprocessing_func=preprocessing_func, patch_height=patch_height, patch_width=patch_width)
    for idx in cls_df.index: #loc[:, 'file']:
        file = cls_df.loc[idx, 'file']
        pred = cls_df.loc[idx, 'image_mean_transform']
        label = np.argmax(pred)
        if isinstance(logger, logging.Logger):
            logger.info(f'''
> File: {file}
- pred: {pred}
- label (argmax(pred)): {label}
            ''')
        # - Create a class directory (if it does not already exists)
        cls_dir = output_dir / f'{label}'
        os.makedirs(cls_dir, exist_ok=True)

        # - Save the image file in the relevant directory
        file_name = file.split('/')[-1]
        shutil.copy(file, cls_dir / f'{file_name}')
