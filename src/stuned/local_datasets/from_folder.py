import torch
import torchvision


# local modules
from .utils import (
    TRAIN_KEY,
    VAL_KEY,
    get_generic_train_eval_dataloaders
)
from .transforms import (
    TRANSFORMS_KEY,
    make_transforms
)


def get_dataloaders_from_folder(
    data_from_folder_config,
    train_batch_size,
    eval_batch_size,
    shuffle_train=True,
    shuffle_eval=False
):

    def clean_splits(splits):
        assert sum(splits.values()) == 1, \
            "Splits for \"from_folder\" dataset do not sum up to one"
        splits_to_pop = []
        for split_name, split_size in splits.items():
            if split_size == 0:
                splits_to_pop.append(split_name)
        for split_to_pop in splits_to_pop:
            splits.pop(split_to_pop)

    assert "root_path" in data_from_folder_config
    assert "splits" in data_from_folder_config
    splits = data_from_folder_config["splits"]
    assert TRAIN_KEY in splits
    assert VAL_KEY in splits

    image_transforms = make_transforms(data_from_folder_config.get(TRANSFORMS_KEY))

    dataset_from_folder = torchvision.datasets.ImageFolder(
        root=data_from_folder_config["root_path"],
        transform=image_transforms
    )

    train_datasets_dict = {}
    eval_datasets_dict = {}

    clean_splits(splits)

    datasets_per_split = torch.utils.data.random_split(dataset_from_folder, splits.values())

    for split_name, dataset in zip(splits.keys(), datasets_per_split):
        if TRAIN_KEY in split_name:
            train_datasets_dict[split_name] = dataset
        else:
            eval_datasets_dict[split_name] = dataset

    train_dataloaders, eval_dataloaders = get_generic_train_eval_dataloaders(
        train_datasets_dict,
        eval_datasets_dict,
        train_batch_size,
        eval_batch_size,
        shuffle_train=shuffle_train,
        shuffle_eval=shuffle_eval
    )

    return train_dataloaders, eval_dataloaders
