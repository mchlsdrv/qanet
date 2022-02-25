import io
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from configs.general_configs import (
    EPSILON,
    MIN_OBJECT_AREA,
)


def get_crop(image: np.ndarray, x: int, y: int, crop_shape: tuple):
    return image[x:x+crop_shape[0], y:y+crop_shape[1]]


def add_channels_dim(image: np.ndarray):
    img = image
    # - If the image is 2D - add the channel dimension
    if len(img.shape) == 2:
        img = np.expand_dims(image, 2)
    return img


def load_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    if len(img.shape) < 3:
        img = add_channels_dim(image=img)
    return img


def get_image_from_figure(figure):
    buffer = io.BytesIO()

    plt.savefig(buffer, format='png')

    plt.close(figure)
    buffer.seek(0)

    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def get_binary_class_label(segmentation: np.ndarray, class_label: int):
    # - Create an empty label, of the similar shape to the segmentation
    bin_lbl = np.zeros_like(segmentation)

    # - Find the indices of the current labels' class in the segmentation label
    seg_px = np.argwhere(segmentation == class_label)

    # - Separate the indices into x and y coordinates
    seg_pix_xs, seg_pix_ys = seg_px[:, 0], seg_px[:, 1]

    # - Mark the entries at the indices that correspond to the label as '1', to produce a binary label
    bin_lbl[(seg_pix_xs, seg_pix_ys)] = 1

    return bin_lbl


def get_largest_label(binary_ground_truth: np.ndarray, multiclass_segmentation: np.ndarray, no_background: bool = True):
    seg_cls_lbl = None

    # (*) As the predicted segmentations' class may differ, we need to infer it from the ground truth label,
    # by looking at the most abundant label in the vicinity of the ground truth label

    # # - Provides the intersection between the ground truth label and the predicted segmentation, which provides all the labels at the same place as the ground truth label,
    # # and the counts the times they appear there. The label which appear the most is considered the representing label of the object
    seg_cls_lbls = binary_ground_truth * multiclass_segmentation  # => as the ground truth label is binary, and the segmentation is multi-class, their multiplication results in the multi-class labels in the place of the ground truth object

    # - If we don't want the background to be included in the classes
    if no_background:
        seg_cls_lbls = seg_cls_lbls[seg_cls_lbls > 0]  # => as '0' will be there, we need to dispose of it, as it is the background

    # - Get the unique classes in the segmentation in the vicinity of the GROUND TRUTH label
    seg_cls_lbls, seg_cls_lbls_cnts = np.unique(seg_cls_lbls, return_counts=True)  # => counting the labels

    # - If the predicted label does not contain any class - skip it, as there is no detections
    if seg_cls_lbls_cnts.any():
        seg_cls_lbl = seg_cls_lbls[np.argmax(seg_cls_lbls_cnts)]  # => here we choose the most abundant label as the class we are looking for in the predictions

    return seg_cls_lbl


def get_seg_measure(ground_truth, segmentation, zero_low_jaccards: bool = True):

    # - Labels in the GROUND TRUTH segmentation to check for
    gt_cls_lbls = np.unique(ground_truth)[1:]  # Exclude the background (the '0')

    # - Jaccard history to compute the mean jaccard at the end
    jaccards = np.array([])

    # - For each class label in the labels from the GROUND TRUTH class labels we check its' segmentation jaccard
    for gt_cls_lbl in gt_cls_lbls:

        # - Prepare binary GROUND TRUTH label for the current class
        gt_bin_cls_lbl = get_binary_class_label(segmentation=ground_truth, class_label=gt_cls_lbl)

        # - If the current GROUND TRUTH label is too small - no need to find its' jaccard, so we just skip it
        if gt_bin_cls_lbl.sum() > MIN_OBJECT_AREA:

            # - Find the class label in the SEGMENTATION which corresponds to the class label in the GROUND TRUTH, by finding
            # the most prevalent label in the SEGMENTATION in the vicinity of the GROUND TRUTH
            seg_cls_lbl = get_largest_label(binary_ground_truth=gt_bin_cls_lbl, multiclass_segmentation=segmentation, no_background=True)

            # - If theres no label in the segmentation - this object was not detected, so add 0. to jaccards, and skip
            if seg_cls_lbl is None:
                jaccards = np.append(jaccards, 0.)  # => add the jaccard coefficient, which in this case is also 0
                continue

            # - Prepare binary SEGMENTATION label for the current class label
            seg_bin_cls_lbl = get_binary_class_label(segmentation=segmentation, class_label=seg_cls_lbl)  # => create an empty label, of the similar shape

            # - Calculate the intersection between the binary GROUND TRUTH and the SEGMENTATION class labels
            I = np.logical_and(gt_bin_cls_lbl, seg_bin_cls_lbl).sum()  # => calculate the intersection of the ground truth with the segmentation

            # - Calculate the union of the binary GROUND TRUTH and the SEGMENTATION class labels
            U = np.logical_or(gt_bin_cls_lbl, seg_bin_cls_lbl).sum()

            # - Calculate the Jaccard coefficient of the current class label
            J = I / (U + EPSILON)

            # - Add the jaccard to history
            jaccards = np.append(jaccards, J)

    # - Return the mean Jaccard of the objects found in the image, or 0 if there's no objects of sufficient area were found
    if zero_low_jaccards:
        jaccards[np.argwhere(jaccards <= .5)] = 0.

    return jaccards.mean() if jaccards.any() else 0.
