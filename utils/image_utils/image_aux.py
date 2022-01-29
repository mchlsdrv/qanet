import io
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
os.chdir('C:/Users/mchls/Desktop/University/PhD/Projects/QANet/qanet')
from configs.general_configs import (
    EPSILON
)


def add_channels_dim(image):
    img = image
    # - If the image is 2D - add the channel dimension
    if len(img.shape) == 2:
        img = np.expand_dims(image, 2)
    return img


def load_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    if len(img.shape) < 3:
        img = np.expand_dims(img, 2)
    return img


def get_image_from_figure(figure):
    buffer = io.BytesIO()

    plt.savefig(buffer, format='png')

    plt.close(figure)
    buffer.seek(0)

    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def get_seg_measure(ground_truth, segmentations):
    labels = np.unique(ground_truth)[1:]  # Exclude the background (the '0')

    number_of_classes = labels.shape[0]

    gt_areas = np.array([])
    I_areas = np.array([])
    jaccards = np.array([])

    for gt_cls in labels:

        # - Prepare the ground truth label
        gt_lbl = np.zeros_like(ground_truth)  # => create an empty label, of the similar shape
        gt_lbl_px = np.argwhere(ground_truth == gt_cls)  # => find the indices of the current labels' class in the ground truth label

        # - If the current ground truth crop does not contain any labels, i.e., is all '0', we have noting to search for, so we skip it
        if gt_lbl_px.shape[0]:
            # - Prepare the ground truth label
            gt_x, gt_y = gt_lbl_px[:, 0], gt_lbl_px[:, 1]  # => separate the indices into x and y coordinates
            gt_lbl[(gt_x, gt_y)] = 1  # => mark the entries at the indices that correspond to the label as '1', to produce a binary label
            gt_areas = np.append(gt_areas, gt_lbl.sum())  # => add the number of pixels of the current label, i.e., the area of the current ground truth label

            # (*) As the predicted segmentations' class may differ, we need to infer it from the ground truth label,
            # by looking at the most abundant label in the vicinity of the ground truth label

            # - Provides the intersection between the ground truth label and the predicted segmentation, which provides all the labels at the same place as the ground truth label,
            # and the counts the times they appear there. The label which appear the most is considered the representing label of the object
            seg_lbls = gt_lbl * segmentations  # => as the ground truth label is binary, and the segmentation is multi-class, their multiplication results in the multi-class labels in the place of the ground truth object
            seg_lbls = seg_lbls[seg_lbls > 0]  # => as '0' will be there, we need to dispose of it, as it is the background
            seg_lbls, seg_lbls_cnts = np.unique(seg_lbls, return_counts=True)  # => counting the labels

            # - If the predicted label does not contain any class - skip it, as there is no detection
            if not seg_lbls_cnts.any():
                I_areas = np.append(I_areas, 0.)  # => add the intersection area, which in this case is 0
                jaccards = np.append(jaccards, 0.)  # => add the jaccard coefficient, which in this case is also 0
                continue

            seg_cls = seg_lbls[np.argmax(seg_lbls_cnts)]  # => here we choose the most abundant label as the class we are looking for in the predictions

            seg_lbl = np.zeros_like(segmentations)  # => create an empty label, of the similar shape
            seg_lbl_px = np.argwhere(segmentations == seg_cls)  # => find the indices of the current labels' class in the segmentation label
            seg_x, seg_y = seg_lbl_px[:, 0], seg_lbl_px[:, 1]  # => separate the indices into x and y coordinates
            seg_lbl[(seg_x, seg_y)] = 1   # => mark the entries at the indices that correspond to the label as '1', to produce a binary label

            # - Calculate the intersection
            I = np.logical_and(gt_lbl, seg_lbl).sum()  # => calculate the intersection of the ground truth with the segmentation
            I_areas = np.append(I_areas, I)  # => add the intersection to history

            # - Calculate the union of the ground truth segmentation with the predicted segmentation
            U = np.logical_or(gt_lbl, seg_lbl).sum()

            # - Calculate the Jaccard coefficient of the current label
            J = I / (U + EPSILON)
            jaccards = np.append(jaccards, J)  # => add the jaccard to history

    # - The detection counts only if the intersection covers at least 0.5 of the objects' area
    # det_objs = I_areas / (gt_areas + EPSILON) > .5

    # - The seg measure is the sum of the jaccards of the detected objects (i.e., the once which overlap by at least 0.5 of objects' area) to the number of different object classes in the crop
    seg_measure = jaccards.sum() / (number_of_classes + EPSILON)
    # seg_measure = jaccards[det_objs].sum() / (number_of_classes + EPSILON)

    return seg_measure
