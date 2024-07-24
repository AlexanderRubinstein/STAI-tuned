import sys
import os
import torch
import torchvision


# local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from stuned.local_datasets.utils import (
    SCALING_FACTOR,
    make_default_data_path,
    make_default_cache_path,
    uniformly_subsample_dataset,
    make_dataset_wrapper_with_index,
    make_caching_dataset
)
from stuned.utility.utils import (
    read_yaml,
    get_project_root_path,
    get_hash
)
from stuned.local_datasets.hugging_face_scripts.imagenet1k_classes import (
    IMAGENET2012_CLASSES
)
from stuned.local_datasets.transforms import (
    TRANSFORMS_KEY
)
from datasets import load_dataset
sys.path.pop(0)


IMAGENET_DEFAULT_RES = (224, 224)
ONLY_VAL_ERROR_MSG = "Only val dataloader is implemented for ImageNet1k"
IMAGENET2012_CLASSES_LIST = [
    (class_id, class_name)
        for class_id, class_name in IMAGENET2012_CLASSES.items()
]
IMAGENET_KEY = "imagenet1k"
IMAGENET_TRAIN_NUM_SAMPLES = 1281167
IMAGENET_VAL_NUM_SAMPLES = 50000
DEFAULT_HUGGING_FACE_ACCESS_TOKEN_PATH \
    = os.path.expanduser("~/.config/hugging_face/hf.yaml")


def make_default_transforms_for_imagenet():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(IMAGENET_DEFAULT_RES)
    ])


def make_torch_tensor_from_image(image):

    # grayscale to RGB
    if len(image.shape) == 2:
        image.unsqueeze_(2)
        image = image.repeat(1, 1, 3)

    return (
        torchvision.transforms.Resize(IMAGENET_DEFAULT_RES)(
            image.permute(2, 0, 1)
        ).to(dtype=torch.float32) / SCALING_FACTOR
    )


def make_transform_for_huggingface(transform):

    def transforms_for_dict(examples):

        examples['image'] = [
            transform(image.convert("RGB")) for image in examples["image"]
        ]

        return examples

    return transforms_for_dict


def get_imagenet_dataset(
    imagenet_config,
    split,
    transform=None,
    num_samples=0,
    subset_indices=None,
    reverse_indices=False
):

    ONLY_VAL = "when \"only_val\" is true"

    # if imagenet_config["only_val"]:
    hf_token_path = imagenet_config.get("hf_token_path")
    if hf_token_path is not None:

        assert num_samples == 0, \
            "Can't specify num_samples for hugging face dataset"

        # where pyarrow files are downloaded
        cache_dir = imagenet_config.get("cache_dir", make_default_cache_path())
        # where splits are created
        data_dir = imagenet_config.get(
            "data_dir",
            os.path.join(make_default_data_path(), f"ImageNet1k_{split}")
        )
        access_token = read_yaml(hf_token_path)["token"]

        if split == "val":
            split = "validation"
        dataset = load_dataset(
            "imagenet-1k",
            split=split,
            cache_dir=cache_dir,
            data_dir=data_dir,
            token=access_token
        )

        dataset.set_format(type='torch', columns=['image', 'label'])

        if transform is None:
            transform = make_default_transforms_for_imagenet()

        dataset = dataset.with_transform(
            make_transform_for_huggingface(transform)
        )

    else:
        assert "path" in imagenet_config
        if transform is None:
            transform = make_default_transforms_for_imagenet()

        dataset = torchvision.datasets.ImageNet(
            root=imagenet_config["path"],
            split=split,
            transform=transform
        )

    if subset_indices is not None:
        assert num_samples == 0, \
            "Can't specify both subset_indices and num_samples"
        if reverse_indices:
            subset_indices = list(set(range(len(dataset))) - set(subset_indices))
        dataset = torch.utils.data.Subset(dataset, subset_indices)

    dataset = uniformly_subsample_dataset(
        dataset,
        num_samples,
        deterministic=True
    )
    return dataset


def collate_fn_hf(examples):
    images = []
    labels = []
    for example in examples:
        images.append(example["image"])
        labels.append(torch.tensor(example["label"], dtype=torch.long))
    return torch.stack(images), torch.stack(labels)


def get_imagenet1k_dataloader(
    imagenet_config,
    split,
    batch_size,
    shuffle,
    transform=None,
    num_samples=0,
    return_index=False,
    subset_indices=None,
    reverse_indices=False,
    num_workers=0
):

    dataset = get_imagenet_dataset(
        imagenet_config,
        split,
        transform=transform,
        num_samples=num_samples,
        subset_indices=subset_indices,
        reverse_indices=reverse_indices
    )

    if imagenet_config.get("caching", False):
        transforms_key = (
            TRANSFORMS_KEY if split == "train" else "eval_transforms"
        )
        cache_path = os.path.join(
            imagenet_config.get(
                "cache_path",
                make_default_cache_path()
            ),
            "ImageNet1k",
            split
        )
        os.makedirs(cache_path, exist_ok=True)
        dataset = make_caching_dataset(
            dataset,
            unique_hash=get_hash(
                    imagenet_config[transforms_key]
                |
                    {"split": split}
                |
                    {"subset": imagenet_config.get("subset_indices", {}).get(split)}
                |
                    {"only_val": imagenet_config.get("only_val")}
            ),
            cache_path=cache_path
        )

    if return_index:
        dataset = make_dataset_wrapper_with_index(dataset)

    use_collate = (imagenet_config.get("hf_token_path") is not None)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=(collate_fn_hf if use_collate else None),
        num_workers=num_workers
    )


def get_imagenet_dataloaders(
    train_batch_size,
    eval_batch_size,
    imagenet_config,
    train_transform=None,
    eval_transform=None,
    return_index=False,
    num_workers=0
):

    trainloader = None
    testloaders = None

    subset_indices = imagenet_config.get("subset_indices", {})

    if train_batch_size > 0:
        assert "train_split" in imagenet_config
        train_split = imagenet_config["train_split"]
        trainloader = get_imagenet1k_dataloader(
            imagenet_config=imagenet_config,
            split=train_split,
            batch_size=train_batch_size,
            shuffle=imagenet_config.get("shuffle_train", True),
            transform=train_transform,
            num_samples=imagenet_config.get("num_train_samples", 0),
            return_index=return_index,
            subset_indices=None,
            reverse_indices=imagenet_config.get("reverse_indices", False),
            num_workers=num_workers
        )

    if eval_batch_size > 0:
        assert "test_splits" in imagenet_config
        testloaders = {
            split: get_imagenet1k_dataloader(
                imagenet_config=imagenet_config,
                split=split,
                batch_size=eval_batch_size,
                shuffle=imagenet_config.get("shuffle_test", False),
                transform=eval_transform,
                num_samples=imagenet_config.get("num_test_samples", 0),
                return_index=return_index,
                subset_indices=None,
                reverse_indices=imagenet_config.get("reverse_indices", False),
                num_workers=num_workers
            ) for split in imagenet_config["test_splits"]
        }

    return trainloader, testloaders
