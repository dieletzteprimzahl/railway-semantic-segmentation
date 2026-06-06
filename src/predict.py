"""
Inference + visualization for railway semantic segmentation.

Loads a trained model and plots Testing Image | Ground Truth | Prediction
for random samples from a validation generator.
"""

import random

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.metrics import MeanIoU

from dataset import train_generator

N_CLASSES = 256
MODEL_PATH = "railsem19_resnet34_20epochs_768patch.hdf5"
VAL_IMG_PATH = "data/val/images"
VAL_MASK_PATH = "data/val/masks"


def evaluate_and_visualize(model_path=MODEL_PATH, n_classes=N_CLASSES):
    model = load_model(model_path, compile=False)

    val_gen = train_generator(
        VAL_IMG_PATH, VAL_MASK_PATH, num_class=n_classes, batch_size=10
    )
    test_image_batch, test_mask_batch = next(val_gen)

    test_mask_argmax = np.argmax(test_mask_batch, axis=3)
    test_pred = model.predict(test_image_batch)
    test_pred_argmax = np.argmax(test_pred, axis=3)

    iou = MeanIoU(num_classes=n_classes)
    iou.update_state(test_pred_argmax, test_mask_argmax)
    print("Mean IoU =", iou.result().numpy())

    img_num = random.randint(0, test_image_batch.shape[0] - 1)
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title("Testing Image")
    plt.imshow(test_image_batch[img_num])
    plt.subplot(232)
    plt.title("Ground Truth")
    plt.imshow(test_mask_argmax[img_num])
    plt.subplot(233)
    plt.title("Prediction")
    plt.imshow(test_pred_argmax[img_num])
    plt.show()


if __name__ == "__main__":
    evaluate_and_visualize()
