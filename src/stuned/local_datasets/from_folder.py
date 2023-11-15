import torch
import torchvision
import copy


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

    def assert_datasets(batch_size, datasets, train_or_eval, splits):
        if batch_size > 0:
            assert len(datasets) > 0, \
                f"There is no split for \"{train_or_eval}\", " \
                f"while \"{train_or_eval}\" batch_size " \
                f"is non-zero ({batch_size})." \
                f"\nsplits: {splits}"

    def clean_splits(splits):
        sum_of_splits = sum(splits.values())
        assert sum_of_splits >= 0 and sum_of_splits <= 1, \
            "Sum of splits for \"from_folder\" dataset should be in [0, 1] range"
        for split_name, split_size in splits.items():
            if split_size is None:
                splits[split_name] = 1 - sum_of_splits
                sum_of_splits = 1
        assert sum_of_splits == 1, \
            "Splits for \"from_folder\" dataset do not sum up to one"
        splits_to_pop = []
        for split_name, split_size in splits.items():
            if split_size == 0:
                splits_to_pop.append(split_name)
        for split_to_pop in splits_to_pop:
            splits.pop(split_to_pop)

    assert "root_path" in data_from_folder_config
    assert "splits" in data_from_folder_config
    splits = copy.deepcopy(data_from_folder_config["splits"])
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

    assert_datasets(train_batch_size, train_datasets_dict, "train", splits)
    assert_datasets(eval_batch_size, eval_datasets_dict, "eval", splits)

    train_dataloaders, eval_dataloaders = get_generic_train_eval_dataloaders(
        train_datasets_dict,
        eval_datasets_dict,
        train_batch_size,
        eval_batch_size,
        shuffle_train=shuffle_train,
        shuffle_eval=shuffle_eval
    )

    return train_dataloaders, eval_dataloaders
