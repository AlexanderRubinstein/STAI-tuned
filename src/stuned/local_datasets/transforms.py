import sys
import os
import torch
import torchvision


# local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utility.utils import (
    raise_unknown,
    parse_name_and_number,
    import_from_string
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

    for transform_type in transforms_list:
        assert transform_type in transforms_config
        transform_name, _ = parse_name_and_number(transform_type)
        if transform_name == "pad":
            result.append(
                torchvision.transforms.Pad(
                    **transforms_config[transform_type]
                )
            )
        elif transform_name == "random_rotate":
            result.append(
                torchvision.transforms.RandomRotation(
                    **transforms_config[transform_type]
                )
            )
        elif transform_name == "random_scale":
            result.append(
                RandomScaleTransform(**transforms_config[transform_type])
            )
        elif transform_name == "random_crop":
            result.append(
                torchvision.transforms.RandomCrop(
                    **transforms_config[transform_type]
                )
            )
        elif transform_name == "random_resized_crop":
            RRC_config = transforms_config[transform_type]
            update_enums_in_config(RRC_config, ["interpolation"])
            result.append(
                torchvision.transforms.RandomResizedCrop(
                    **RRC_config
                )
            )
        elif transform_name == "from_class":

            from_class_config = transforms_config[transform_type]
            assert "class" in from_class_config
            class_ctor = import_from_string(from_class_config["class"])
            transform_kwargs = from_class_config.get("kwargs", {})
            importable_transform_kwargs = from_class_config.get("kwargs_to_import", {})

            update_enums_in_config(
                importable_transform_kwargs,
                importable_transform_kwargs.keys()
            )
            result.append(
                class_ctor(
                    **(transform_kwargs | importable_transform_kwargs)
                )
            )

        else:
            raise_unknown("transform name", transform_name, transforms_config)

    return torchvision.transforms.Compose(result)


def update_enums_in_config(config, enums, nested_attrs_depth=2):
    for enum in enums:
        if enum in config:
            config[enum] = import_from_string(
                config[enum],
                nested_attrs_depth=nested_attrs_depth
            )


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
