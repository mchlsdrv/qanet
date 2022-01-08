import os
import io
import numpy as np
import pandas as pd
import pathlib
import tensorflow as tf
import cv2
from utils.image_utils import preprocessing_funcs
import matplotlib.pyplot as plt


def get_random_crop(images, crop_shape):
    # 4) Randomly crop the image
    rnd_pt = tf.random.uniform(shape=(1, 2))
    img = tf.image.crop_and_resize(
        img,
        boxes=[[rnd_pt[1], rnd_pt[0], rnd_pt[1] + crop_size[1], rnd_pt[0] + crop_size[0]]]
    )
    img = tf.cast(img, tf.float32)


    return img, label


def get_patch_df(image_file, preprocessing_func,  patch_height, patch_width):
    assert image_file.is_file(), f'No file \'{image_file}\' was found!'

    img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, axis=-1)
    img = preprocessing_func(img)
    df = pd.DataFrame(columns=['file', 'image'])
    img_h, img_w, _ = img.shape
    for h in range(0, img_h, patch_height):
        for w in range(0, img_w, patch_width):
            patch = img[h:h+patch_height, w:w+patch_width, :]
            if patch.shape[0] == patch_height and patch.shape[1] == patch_width:
                df = df.append(dict(file=str(image_file), image=patch), ignore_index=True)
    return df


def get_mean_image_transforms(images_root_dir, model, preprocessing_func, patch_height, patch_width):
    df = pd.DataFrame(columns=['file', 'image_mean_transform'])
    for root, dirs, files in os.walk(images_root_dir):
        for file in files:

            # get the patches
            patches_df = get_patch_df(image_file=pathlib.Path(f'{root}/{file}'), preprocessing_func=preprocessing_func, patch_height=patch_height, patch_width=patch_width)

            # get the mean patch transform
            patch_transforms = list()
            for patch in patches_df.loc[:, 'image'].values:
                patch_transforms.append(model(np.expand_dims(patch, axis=0)) if len(patch.shape) < 4 else model(patch))
            patch_transforms = np.array(patch_transforms)
            image_mean_transform = patch_transforms.mean(axis=0)[0, :]
            df = df.append(
                {
                    'file': f'{root}/{file}',
                    'image_mean_transform': image_mean_transform
                },
                ignore_index=True
            )
    return df


def get_patch_transforms(images_root_dir, model, preprocessing_func, patch_height, patch_width):
    df = pd.DataFrame(columns=['file', 'image'])
    for root, dirs, files in os.walk(images_root_dir):
        for file in files:
            df = df.append(get_patch_df(image_file=pathlib.Path(f'{root}/{file}'), preprocessing_func=preprocessing_func, patch_height=patch_height, patch_width=patch_width), ignore_index=True)
    df.loc[:, 'patch_transform'] = df.loc[:, 'image'].apply(lambda x: model(np.expand_dims(x, axis=0))[0].numpy() if len(x.shape) < 4 else model(x)[0].numpy())
    df = df.loc[:, ['file', 'patch_transform']]
    return df


def get_image_from_figure(figure):
    buffer = io.BytesIO()

    plt.savefig(buffer, format='png')

    plt.close(figure)
    buffer.seek(0)

    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image
