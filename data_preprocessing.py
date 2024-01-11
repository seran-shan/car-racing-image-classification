"""
This module contains functions for loading and preprocessing images.
"""
import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical


def load_and_preprocess_images(directory):
    """
    Loads and preprocesses images from the given directory.

    Parameters
    ----------
    directory : str
        The directory to load the images from.

    Returns
    -------
    images : numpy.ndarray
        The images.
    labels : numpy.ndarray
        The labels.
    """
    images = []
    labels = []

    for label in range(5):  # Assuming 5 classes labeled as 0-4
        label_directory = os.path.join(directory, str(label))
        for file in os.listdir(label_directory):
            img_path = os.path.join(label_directory, file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0  # Normalization
            images.append(img)
            labels.append(label)

    return np.array(images), to_categorical(labels, num_classes=5)
