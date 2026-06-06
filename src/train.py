"""
Training entry point for railway semantic segmentation.

Trains a ResNet34 U-Net on patched RailSem19 data and saves the model.
Run after the dataset has been patchified and split (see dataset.py).
"""

import os

from model import build_model
from dataset import train_generator

# ---- Configuration ----------------------------------------------------------
DATA_ROOT = "data"  # output of split_dataset()
TRAIN_IMG_PATH = os.path.join(DATA_ROOT, "train", "images")
TRAIN_MASK_PATH = os.path.join(DATA_ROOT, "train", "masks")
VAL_IMG_PATH = os.path.join(DATA_ROOT, "val", "images")
VAL_MASK_PATH = os.path.join(DATA_ROOT, "val", "masks")

N_CLASSES = 256       # see note in model.build_model
BATCH_SIZE = 10
EPOCHS = 20
IMG_HEIGHT = 768
IMG_WIDTH = 768
IMG_CHANNELS = 3
MODEL_OUT = "railsem19_resnet34_20epochs_768patch.hdf5"


def main():
    train_gen = train_generator(
        TRAIN_IMG_PATH, TRAIN_MASK_PATH, num_class=N_CLASSES, batch_size=BATCH_SIZE
    )
    val_gen = train_generator(
        VAL_IMG_PATH, VAL_MASK_PATH, num_class=N_CLASSES, batch_size=BATCH_SIZE
    )

    num_train = len(os.listdir(os.path.join(TRAIN_IMG_PATH, "train")))
    num_val = len(os.listdir(os.path.join(VAL_IMG_PATH, "val")))
    steps_per_epoch = num_train // BATCH_SIZE
    val_steps = num_val // BATCH_SIZE

    model = build_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, n_classes=N_CLASSES)
    model.summary()

    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        verbose=1,
        validation_data=val_gen,
        validation_steps=val_steps,
    )

    model.save(MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
