"""
Data pipeline for railway semantic segmentation.

Steps:
    1. Patchify full-size images/masks into 768x768 tiles.
    2. Split into train/validation folders.
    3. Build augmenting data generators for training.
"""

import os
import cv2
import numpy as np
from PIL import Image
from patchify import patchify
import splitfolders
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import get_preprocessing

PATCH_SIZE = 768
scaler = MinMaxScaler()


def patchify_images(img_dir, out_dir, patch_size=PATCH_SIZE):
    """Crop each large image to a size divisible by patch_size, then split
    into non-overlapping patch_size x patch_size RGB tiles.

    Resizing is intentionally avoided — it distorts object scale and hurts
    segmentation quality.
    """
    os.makedirs(out_dir, exist_ok=True)
    for path, _, _ in os.walk(img_dir):
        images = sorted(os.listdir(path))
        for image_name in images:
            if not image_name.endswith(".jpg"):
                continue
            image = cv2.imread(os.path.join(path, image_name), 1)
            size_x = (image.shape[1] // patch_size) * patch_size
            size_y = (image.shape[0] // patch_size) * patch_size
            image = Image.fromarray(image).crop((0, 0, size_x, size_y))
            image = np.array(image)

            patches = patchify(image, (patch_size, patch_size, 3), step=patch_size)
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    single = patches[i, j, :, :][0]
                    cv2.imwrite(
                        os.path.join(out_dir, f"{image_name}patch_{i}{j}.jpg"),
                        single,
                    )


def patchify_masks(mask_dir, out_dir, patch_size=PATCH_SIZE):
    """Same as patchify_images but for grayscale masks (single channel).

    Masks must NOT be resized/interpolated — pixel values are class labels.
    """
    os.makedirs(out_dir, exist_ok=True)
    for path, _, _ in os.walk(mask_dir):
        masks = sorted(os.listdir(path))
        for mask_name in masks:
            if not mask_name.endswith(".png"):
                continue
            mask = cv2.imread(os.path.join(path, mask_name), 0)
            size_x = (mask.shape[1] // patch_size) * patch_size
            size_y = (mask.shape[0] // patch_size) * patch_size
            mask = Image.fromarray(mask).crop((0, 0, size_x, size_y))
            mask = np.array(mask)

            patches = patchify(mask, (patch_size, patch_size), step=patch_size)
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    single = patches[i, j, :, :]
                    cv2.imwrite(
                        os.path.join(out_dir, f"{mask_name}patch_{i}{j}.png"),
                        single,
                    )


def split_dataset(input_folder, output_folder="data", ratio=(0.8, 0.2), seed=1337):
    """Split patched images/masks into train/val folders."""
    splitfolders.ratio(
        input_folder, output=output_folder, seed=seed, ratio=ratio, group_prefix=None
    )


def preprocess_data(img, mask, num_class, preprocess_input):
    """Scale images, apply backbone preprocessing, one-hot encode masks."""
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    img = preprocess_input(img)
    mask = to_categorical(mask, num_class)
    return img, mask


def train_generator(train_img_path, train_mask_path, num_class,
                    batch_size=10, seed=24, backbone="resnet34"):
    """Yield augmented (image, mask) batches read directly from disk.

    Only flips are used — no rotation/zoom, to avoid interpolating mask
    label values.
    """
    preprocess_input = get_preprocessing(backbone)

    aug_args = dict(horizontal_flip=True, vertical_flip=True, fill_mode="reflect")
    image_datagen = ImageDataGenerator(**aug_args)
    mask_datagen = ImageDataGenerator(**aug_args)

    image_generator = image_datagen.flow_from_directory(
        train_img_path, class_mode=None, batch_size=batch_size, seed=seed
    )
    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path, class_mode=None, color_mode="grayscale",
        batch_size=batch_size, seed=seed
    )

    for img, mask in zip(image_generator, mask_generator):
        yield preprocess_data(img, mask, num_class, preprocess_input)
