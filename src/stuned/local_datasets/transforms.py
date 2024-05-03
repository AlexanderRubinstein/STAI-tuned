import sys
import os
import torch
import torchvision


# local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utility.utils import (
    raise_unknown
)
from utility.imports import (
    FROM_CLASS_KEY,
    make_from_class_ctor
)
from local_datasets.utils import (
    randomly_subsample_indices_uniformly
)
sys.path.pop(0)


TRANSFORMS_KEY = "transforms"
DEFAULT_MEAN_IN = [0.485, 0.456, 0.406]
DEFAULT_STD_IN = [0.229, 0.224, 0.225]
DEFAULT_SIZE_IN = 224
DEFAULT_RESIZE_IN = 256
DEFAULT_SCALE_IN = (0.08, 1.0)
DEFAULT_RATIO_IN = (0.75, 1.3333333333333333)


def make_normalization_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            DEFAULT_MEAN_IN,
            DEFAULT_STD_IN
        )
    ])


def make_default_train_transforms_imagenet():
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(
                scale=DEFAULT_SCALE_IN,
                ratio=DEFAULT_RATIO_IN,
                size=DEFAULT_SIZE_IN

            ),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
            make_normalization_transforms()
        ]
    )


def make_default_test_transforms_imagenet():
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(DEFAULT_RESIZE_IN),
            torchvision.transforms.CenterCrop(DEFAULT_SIZE_IN),
            torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
            make_normalization_transforms()
        ]
    )


def make_transforms(transforms_config):

    if transforms_config is None:
        return None

    if transforms_config == "ImageNetEval":
        return make_default_test_transforms_imagenet()
    elif transforms_config == "ImageNetTrain":
        return make_default_train_transforms_imagenet()

    transforms_list = transforms_config.get("transforms_list", [])

    if len(transforms_list) == 0:
        return None

    result = []

    for transform_name in transforms_list:
        assert transform_name in transforms_config
        specific_transform_config = transforms_config[transform_name]
        if transform_name.startswith(FROM_CLASS_KEY):
            result.append(
                make_from_class_ctor(specific_transform_config)
            )

        elif transform_name.startswith("random_scale"):
            result.append(
                RandomScaleTransform(**specific_transform_config)
            )

        else:
            raise_unknown("transform name", transform_name, transforms_config)

    return torchvision.transforms.Compose(result)


class RandomScaleTransform(torch.nn.Module):

    def __init__(self, scales):
        super(RandomScaleTransform, self).__init__()
        self.scales = scales

    def forward(self, x):
        x_shape = x.shape
        assert len(x_shape) == 4
        rand_n = randomly_subsample_indices_uniformly(
            len(self.scales),
            1
        ).item()
        scale = self.scales[rand_n]

        scale_shape = x_shape[:2] + tuple(
            [int(el * scale) for el in x_shape[2:]]
        )

        return torchvision.transforms.Resize(scale_shape[2:])(x)
