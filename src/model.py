"""
Model definition for railway semantic segmentation.

ResNet34 encoder (pretrained on ImageNet) + U-Net decoder via the
`segmentation_models` library, plus custom Jaccard loss and Dice metric.
"""

import segmentation_models as sm
from tensorflow.keras.optimizers import Adam
from focal_loss import BinaryFocalLoss
from keras import backend as K

sm.set_framework("tf.keras")
sm.framework()


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """Jaccard (IoU) distance loss. Useful for unbalanced datasets.

    Jaccard = |X & Y| / (|X| + |Y| - |X & Y|)

    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    """
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    sum_ = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def dice_metric(y_pred, y_true):
    """Dice coefficient — overlap between prediction and ground truth.

    Returns a value in [0, 1]; closer to 1 means higher similarity.
    """
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    return 2 * intersection / union


def build_model(img_height, img_width, img_channels=3, n_classes=256,
                backbone="resnet34", lr=1e-3):
    """Build and compile a ResNet34 U-Net for semantic segmentation.

    Args:
        img_height, img_width, img_channels: input image dimensions.
        n_classes: number of output channels. NOTE: RailSem19 defines 20
            semantic classes, but masks are stored as 8-bit grayscale, so
            256 channels are used to map directly onto raw pixel values.
            Only ~20 channels carry real labels.
        backbone: encoder architecture (pretrained on ImageNet).
        lr: learning rate for the Adam optimizer.

    Returns:
        A compiled tf.keras Model.
    """
    model = sm.Unet(
        backbone,
        encoder_weights="imagenet",
        input_shape=(img_height, img_width, img_channels),
        classes=n_classes,
        activation="softmax",
    )
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=BinaryFocalLoss(gamma=2),
        metrics=[dice_metric],
    )
    return model


def get_preprocessing(backbone="resnet34"):
    """Return the backbone-specific input preprocessing function."""
    return sm.get_preprocessing(backbone)
