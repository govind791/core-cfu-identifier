from __future__ import annotations
import albumentations as A


def get_train_transform():
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=1.0
            ),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    ],
    keypoint_params=A.KeypointParams(
        format="xy",
        remove_invisible=True,
    ))


def get_val_transform():
    return A.Compose([
        A.Resize(512, 512),
    ],
    keypoint_params=A.KeypointParams(
        format="xy",
        remove_invisible=True,
    ))