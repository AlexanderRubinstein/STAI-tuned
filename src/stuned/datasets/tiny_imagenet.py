import os
import tempfile
from torchvision import (
    datasets,
    transforms as T
)
from torch.utils.data import DataLoader
import torch


# local modules
from ..datasets.utils import (
    fetch_data,
    make_contents_as_in_subfolder,
    remove_file_or_folder,
    move_folder_contents
)
from ..utility.logger import (
    make_logger
)
from ..utility.utils import (
    SYSTEM_PLATFORM,
    runcmd,
    randomly_subsample_indices_uniformly,
    deterministically_subsample_indices_uniformly
)


TINY_IMAGENET_NAME = "tiny-imagenet-200"
TINY_IMAGENET_URL \
    = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_NCLASSES = 200
TINY_IMAGENET_TRAIN_PER_CLASS = 500
TINY_IMAGENET_VAL_PER_CLASS = 50
TINY_IMAGENET_TEST_PER_CLASS = 50
TINY_IMAGENET_NUM_TEST_SAMPLES = 10000
TINY_IMAGENET_TRAIN_MEAN = [0.485, 0.456, 0.406]
TINY_IMAGENET_TRAIN_STD = [0.229, 0.224, 0.225]
TINY_IMAGENET_DEFAULT_MEAN = [0.5, 0.5, 0.5]
TINY_IMAGENET_DEFAULT_STD = [0.5, 0.5, 0.5]
TINY_IMAGENET_RESIZE = 256
TINY_IMAGENET_CROP = 224


def assert_tiny_imagenet_split_folder(
    split_dir,
    samples_per_class
):
    split_dir_contents = os.listdir(split_dir)
    if len(split_dir_contents) != TINY_IMAGENET_NCLASSES:
        return False
    for dir in split_dir_contents:
        images_subdir = os.path.join(split_dir, dir, "images")
        if not (os.path.exists(images_subdir)):
            return False
        if len(os.listdir(images_subdir)) != samples_per_class:
            return False
    return True


def assert_tiny_imagenet(dataset_path):
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")
    test_dir = os.path.join(dataset_path, "test")
    if not (
        os.path.exists(train_dir)
            and os.path.exists(test_dir)
            and os.path.exists(val_dir)
    ):
        return False
    return (
        assert_tiny_imagenet_split_folder(
            train_dir,
            TINY_IMAGENET_TRAIN_PER_CLASS
        )
        and assert_tiny_imagenet_split_folder(
            val_dir,
            TINY_IMAGENET_VAL_PER_CLASS
        )
        and assert_tiny_imagenet_test(test_dir)
    )


def reorganize_tiny_imagenet_val(dataset_path):
    # Create separate validation subfolders for the validation
    # images based on their labels
    # indicated in the val_annotations txt file
    source_folder = os.path.join(dataset_path, "val")
    val_img_dir = os.path.join(source_folder, "images")
    annotations_file = os.path.join(
        source_folder,
        "val_annotations.txt"
    )

    fp = open(annotations_file, 'r')
    data = fp.readlines()

    # Create dictionary to store img filename (word 0)
    # and corresponding label (word 1)
    # for every line in the txt file (as key value pair)
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create subfolders (if not present) for validation images
    # based on label,
    # and move images into the respective folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(
            os.path.dirname(val_img_dir),
            folder,
            "images"
        ))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(val_img_dir, img)):
            os.rename(
                os.path.join(val_img_dir, img),
                os.path.join(newpath, img)
            )
    remove_file_or_folder(val_img_dir)
    remove_file_or_folder(annotations_file)


def do_fetch_tiny_imagenet(
    dataset_name,
    dataset_path,
    reorganize_val=True,
    logger=make_logger()
):

    if SYSTEM_PLATFORM == "linux" or SYSTEM_PLATFORM == "linux2":

        logger.log(
            "Downloading \"{}\" into {}..".format(
                dataset_name,
                dataset_path
            ),
            auto_newline=True
        )

        runcmd(
            "wget {} -O {}.zip".format(
                TINY_IMAGENET_URL,
                dataset_path
            ),
            verbose=True,
            logger=logger
        )

        with tempfile.TemporaryDirectory(
            dir=os.path.dirname(dataset_path)
        ) as tmp_dir:
            logger.log(
                "Unzipping \"{}\" into {}.. "
                "(it might take up to 20 mins)".format(
                    dataset_name,
                    dataset_path
                ),
                auto_newline=True
            )
            runcmd(
                "unzip {}.zip -d {}".format(
                    dataset_path,
                    tmp_dir
                ),
                logger=logger
            )

            extracted_folders = os.listdir(tmp_dir)
            assert len(extracted_folders) == 1
            name_of_extracted_folder = os.path.join(
                tmp_dir,
                extracted_folders[0]
            )
            move_folder_contents(
                name_of_extracted_folder,
                dataset_path
            )

        remove_file_or_folder("{}.zip".format(dataset_path))

        if reorganize_val:
            reorganize_tiny_imagenet_val(dataset_path)
        logger.log(
            "\"{}\" is in {} now!".format(
                dataset_name,
                dataset_path
            ),
            auto_newline=True
        )
    else:
        raise Exception(
            "Download is not implemented for \"{}\" system"
            ", please download manually to {} from {}".format(
                SYSTEM_PLATFORM,
                dataset_path,
                TINY_IMAGENET_URL
            )
        )


def assert_tiny_imagenet_test(test_dataset_path):
    images_path = os.path.join(test_dataset_path, "images")
    return (
        os.path.exists(images_path)
            and len(
                os.listdir(
                    images_path
                )
            )
                == TINY_IMAGENET_NUM_TEST_SAMPLES
    )


def generate_tiny_imagenet_dataset(data, transform):
    return datasets.ImageFolder(
        data,
        transform=transform if transform else T.ToTensor()
    )


def tiny_imagenet_transform(
    to_normalize,
    random_tranforms=True
):
    if to_normalize:
        mean = TINY_IMAGENET_TRAIN_MEAN
        std = TINY_IMAGENET_TRAIN_STD
    else:
        mean = TINY_IMAGENET_DEFAULT_MEAN
        std = TINY_IMAGENET_DEFAULT_STD
    transforms_list = [
        T.Resize(TINY_IMAGENET_RESIZE),
        T.CenterCrop(TINY_IMAGENET_CROP),
    ]
    if random_tranforms:
        transforms_list += [T.RandomHorizontalFlip()]
    transforms_list += [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ]
    transform = T.Compose(transforms_list)

    return transform


def get_tiny_imagenet_dataloaders(
    tiny_imagenet_config,
    params_config,
    logger=make_logger()
):

    dataset_path = os.path.join(
        tiny_imagenet_config["path"],
        TINY_IMAGENET_NAME
    )

    fetch_data(
        dataset_path=dataset_path,
        assert_folder_func=assert_tiny_imagenet,
        do_fetch_data_func=do_fetch_tiny_imagenet,
        logger=logger
    )

    total_num_samples = tiny_imagenet_config["total_number_of_samples"]
    if total_num_samples > 0:
        num_train_samples = max(
            1,
            round(tiny_imagenet_config["train_val_split"] * total_num_samples)
        )
        num_val_samples = total_num_samples - num_train_samples
    else:
        num_train_samples = 0
        num_val_samples = 0
    to_normalize = tiny_imagenet_config["normalize"]
    train_dataloader = generate_tiny_imagenet_dataloader(
        params_config,
        os.path.join(dataset_path, "train"),
        "train",
        tiny_imagenet_transform(
            to_normalize,
            random_tranforms=True
        ),
        num_train_samples,
        logger=logger
    )
    val_dataloader = generate_tiny_imagenet_dataloader(
        params_config,
        os.path.join(dataset_path, "val"),
        "val",
        tiny_imagenet_transform(
            to_normalize,
            random_tranforms=False
        ),
        num_val_samples,
        logger=logger
    )
    return train_dataloader, {"val": val_dataloader}


def generate_tiny_imagenet_dataloader(
    params_config,
    data,
    name,
    transform,
    num_samples=0,
    logger=make_logger()
):

    if data is None:
        return None

    is_train = (name == "train")

    logger.log(
        "Applying the following transform to \"{}\" data: \n{}".format(
            name,
            transform
        )
    )
    dataset = generate_tiny_imagenet_dataset(data, transform)
    if num_samples > 0:
        if is_train:
            indices = randomly_subsample_indices_uniformly(
                len(dataset),
                num_samples
            )
        else:
            indices \
                = deterministically_subsample_indices_uniformly(
                    len(dataset),
                    num_samples
                )
        dataset = torch.utils.data.Subset(dataset, indices)
    batch_size = (
        params_config["train"]["batch_size"]
            if is_train
            else params_config["eval"]["batch_size"]
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train
    )


def do_fetch_tiny_imagenet_test_for_manifold_projector(
    dataset_name,
    dataset_path,
    logger=make_logger()
):

    do_fetch_tiny_imagenet(
        dataset_name,
        dataset_path,
        reorganize_val=False,
        logger=logger
    )
    make_contents_as_in_subfolder(dataset_path, "test")


def get_tiny_imagenet_test_projector_dataset(
    manifold_projector_data_name,
    tiny_imagenet_test_config,
    n_images,
    logger=make_logger()
):

    dataset_path = os.path.join(
        tiny_imagenet_test_config["path"],
        manifold_projector_data_name
    )

    fetch_data(
        dataset_path=dataset_path,
        assert_folder_func=assert_tiny_imagenet_test,
        do_fetch_data_func
            =do_fetch_tiny_imagenet_test_for_manifold_projector,
        logger=logger
    )

    imagenet_transform = tiny_imagenet_transform(
        tiny_imagenet_test_config["normalize"],
        logger
    )

    # flatten images so that projector has 1-d inputs
    final_transform = T.Compose(
        [
            imagenet_transform,
            T.Lambda(lambda x: torch.flatten(x))
        ]
    )
    dataset = generate_tiny_imagenet_dataset(
        dataset_path,
        final_transform
    )

    if n_images == 0:
        return dataset
    else:
        return torch.utils.data.Subset(
            dataset,
            [i for i in range(n_images)]
        )
