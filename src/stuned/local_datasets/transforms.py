import sys
import os
import torch
import torchvision
import copy


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


def make_transforms(transforms_config):

    if transforms_config is None:
        return None

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
