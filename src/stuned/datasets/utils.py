import os
from torch.utils.data import (
    DataLoader,
    SubsetRandomSampler
)
import shutil
import gdown
import requests
import random
import pickle
import traceback
from typing import (
    Union,
    Dict,
    List,
    Any,
    Callable
)
import torch


# local modules
from ..utility.logger import make_logger
from ..utility.utils import (
    remove_filename_extension,
    remove_file_or_folder,
    remove_all_but_subdirs,
    move_folder_contents,
    compute_file_hash,
    get_hash,
    log_or_print,
    get_stuned_root_path,
    prepare_for_pickling,
    append_dict,
    show_images
)


CHECK_FILE_HASH = False


EMPTY_URL = "¯\_(ツ)_/¯"
YANDEX_API_ENDPOINT = "https://cloud-api.yandex.net/v1/disk" \
    + "/public/resources/download?public_key={}"


H5_EXTENSION = ".h5"


DEFAULT_CACHE_PATH = os.path.join(get_stuned_root_path(), "cache")


def fetch_data(
    dataset_path,
    assert_folder_func,
    do_fetch_data_func,
    logger=make_logger(),
    dataset_path_is_folder=True
):
    """
    Makes sure that dataset is under the <dataset_path> path
    """
    dataset_name = remove_filename_extension(
        os.path.basename(dataset_path),
        must_have_extension=False
    )
    logger.log(
        "Verifying \"{}\" dataset..".format(dataset_name)
    )
    if not assert_folder_func(dataset_path):
        logger.log(
            "Could not verify \"{}\" dataset in {}".format(
                dataset_name,
                dataset_path
            ),
            auto_newline=True
        )
        if os.path.exists(dataset_path):
            object_type \
                = "folder" \
                    if dataset_path_is_folder \
                    else "file"
            logger.log(
                "Removing existing {} {}..".format(
                    object_type,
                    dataset_path
                )
            )
            remove_file_or_folder(dataset_path)
        os.makedirs(
            dataset_path
                if dataset_path_is_folder
                else os.path.dirname(dataset_path),
            exist_ok=True
        )
        do_fetch_data_func(
            dataset_name,
            dataset_path,
            logger=logger
        )
        logger.log(
            "Verifying \"{}\" dataset "
            "downloaded into {}..".format(
                dataset_name,
                dataset_path
            ),
            auto_newline=True
        )
        if not assert_folder_func(dataset_path):
            raise Exception(
                "Downloaded data is not what was expected."
            )
    else:
        logger.log(
            "\"{}\" is already in {}!".format(
                dataset_name,
                dataset_path
            ),
            auto_newline=True
        )


def convert_dataset_to_tensor(dataset):
    data_loader = DataLoader(dataset, batch_size=len(dataset))
    return next(iter(data_loader))


def convert_dataset_to_tensor_generator(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size)


def make_contents_as_in_subfolder(
    src_folder,
    subdir_name
):

    remove_all_but_subdirs(src_folder, [subdir_name])
    folder_to_keep = os.path.join(src_folder, subdir_name)
    move_folder_contents(folder_to_keep, src_folder)
    shutil.rmtree(folder_to_keep)


def do_fetch_dataset_as_file(url):

    def do_fetch_file_by_url(
        dataset_name,
        dataset_path,
        logger=make_logger()
    ):
        if url == EMPTY_URL:
            NotImplementedError()
        logger.log(
            "Downloading \"{}\" into {}..".format(
                dataset_name,
                dataset_path
            ),
            auto_newline=True
        )
        download_file_by_link(url, dataset_path)

    return do_fetch_file_by_url


def download_file_by_link(url, path):
    if "google" in url:
        gdown.download(url, path, quiet=True)
    elif "yandex" in url:
        download_file_by_yadisk_link(url, path)
    else:
        raise Exception(
            "Can not download by url which is not from Yandex "
            "or Google drive"
        )


def download_file_by_yadisk_link(sharing_link, filename):

    def _get_real_direct_link(sharing_link):
        pk_request = requests.get(
            YANDEX_API_ENDPOINT.format(sharing_link)
        )
        return pk_request.json().get('href')

    direct_link = _get_real_direct_link(sharing_link)
    if direct_link:
        download = requests.get(direct_link)
        with open(filename, 'wb') as out_file:
            out_file.write(download.content)
    else:
        raise Exception(
            "Can not download file from {}".format(
                sharing_link
            )
        )


def assert_dataset_as_file(file_hash):

    def assert_file_by_hash(dataset_path):

        path_exists = os.path.exists(dataset_path)

        file_hash_is_correct = True
        if path_exists and CHECK_FILE_HASH:
            file_hash_is_correct \
                = (compute_file_hash(dataset_path) == file_hash)

        return path_exists and file_hash_is_correct

    return assert_file_by_hash


def randomly_subsampled_dataloader(dataloader, num_samples, batch_size=None):

    assert isinstance(dataloader, DataLoader)

    subsampled_indices = random.sample(
        list(range(len(dataloader.dataset))),
        num_samples
    )

    dataloader_init_args = get_dataloader_init_args_from_existing_dataloader(
        dataloader
    )

    dataloader_init_args["sampler"] = SubsetRandomSampler(
        subsampled_indices
    )
    if batch_size:
        dataloader_init_args["batch_size"] = batch_size

    new_dataloader = DataLoader(**dataloader_init_args)

    return new_dataloader


def get_dataloader_init_args_from_existing_dataloader(
    dataloader,
    copy_batch_sampler=False
):
    dataloader_init_args = {
        "dataset": dataloader.dataset,
        "num_workers": dataloader.num_workers,
        "collate_fn": dataloader.collate_fn,
        "pin_memory": dataloader.pin_memory,
        "timeout": dataloader.timeout,
        "worker_init_fn": dataloader.worker_init_fn,
        "prefetch_factor": dataloader.prefetch_factor,
        "persistent_workers": dataloader.persistent_workers
    }
    if copy_batch_sampler:
        dataloader_init_args["batch_sampler"] = dataloader.batch_sampler
    else:
        dataloader_init_args["sampler"] = dataloader.sampler
        dataloader_init_args["batch_size"] = dataloader.batch_size
        dataloader_init_args["drop_last"] = dataloader.drop_last
    return dataloader_init_args


def make_or_load_from_cache(
    object_name,
    object_config,
    make_func,
    load_func,
    cache_path,
    forward_cache_path,
    logger=make_logger()
):

    if cache_path is None:
        cache_fullpath = None
    else:
        os.makedirs(cache_path, exist_ok=True)
        cache_fullpath = os.path.join(
            cache_path,
            "{}_{}.pkl".format(
                object_name,
                get_hash(object_config)
            )
        )

    if cache_fullpath and os.path.exists(cache_fullpath):
        log_or_print(
            "Loading cached {} from {}".format(
                object_name,
                cache_fullpath
            ),
            logger=logger,
            auto_newline=True
        )

        try:
            result = load_func(cache_fullpath)
            # TODO(Alex | 22.02.2023) Move to load_func like in UT-TML repo
            if hasattr(result, "logger"):
                result.logger = logger
            return result
        except:
            logger.error("Could not load object from {}\nReason:\n{}".format(
                cache_fullpath,
                traceback.format_exc())
            )

    if forward_cache_path:
        result = make_func(
            object_config,
            cache_path=cache_path,
            logger=logger
        )
    else:
        result = make_func(
            object_config,
            logger=logger
        )

    if cache_fullpath:
        try:
            # TODO(Alex | 22.02.2023) Move to save_func like in UT-TML repo
            prepare_for_pickling(result)
            pickle.dump(result, open(cache_fullpath, "wb"))
            log_or_print(
                "Saved cached {} into {}".format(
                    object_name,
                    cache_fullpath
                ),
                logger=logger,
                auto_newline=True
            )
        except OSError:
            log_or_print(
                "Could not save cached {} to {}. "
                "Reason: \n{} \nContinuing without saving it.".format(
                    object_name,
                    cache_fullpath,
                    traceback.format_exc()
                ),
                logger=logger,
                auto_newline=True
            )
    return result


def show_dataloader_first_batch(
    dataloader: torch.utils.data.DataLoader,
    label_names: List[str]
):
    """
    Plots first batch of a dataloader
    using local function "show_images_batch".
    Args:
        dataloader (torch.utils.data.DataLoader): a dataloader
            which first batch is shown.
        label_names (List[str]): list of names for each label.
            For example, if a dataloader generates
            tuple (input, List[label_0, ... label_k]),
            label_names will be List[label_0_name, ..., label_k_name].
    """

    iterator_element = next(iter(dataloader))
    assert len(iterator_element) > 1
    if len(iterator_element) == 2:
        images_batch, labels_batch = iterator_element
    else:
        images_batch = iterator_element[0]
        labels_batch = list(iterator_element[1:])

    images_batch = images_batch.cpu()

    if isinstance(labels_batch, list):
        for i in range(len(labels_batch)):
            labels_batch[i] = labels_batch[i].cpu()
    else:
        labels_batch = labels_batch.cpu()

    labels_batch = make_named_labels_batch(label_names, labels_batch)

    show_images_batch(images_batch, labels_batch)


def show_images_batch(
    images_batch: torch.tensor,
    label_batches: Union[torch.tensor, Dict[str, torch.tensor]] = None
):
    """
    Shows a batch of images as a square image grid.
    If <label_batches> is provided, each image title consists of label names
    and their corresponding values.
    Args:
        images_batch (torch.tensor): of images batch.
        label_batches (Union[torch.tensor, Dict[str, torch.tensor]], optional):
            information about labels that is either a label batch
            or a dict that maps label name to the corresponding label batch.
    """


    images_list = []

    images_batch = images_batch.cpu()

    label_lists = None if label_batches is None else {}

    n_images = images_batch.shape[0]

    if label_batches is not None:

        if not isinstance(label_batches, dict):
            label_batches = {"label": label_batches}

        for label_batch in label_batches.values():
            assert label_batch.shape[0] == n_images

    for i in range(n_images):
        images_list.append(images_batch[i])
        if label_lists is not None:
            for label_name, label_batch in label_batches.items():
                append_dict(
                    label_lists,
                    {label_name: label_batch[i].item()},
                    allow_new_keys=True
                )

    show_images(images_list, label_lists)


def make_named_labels_batch(label_names, labels_batch):
    if isinstance(labels_batch, list):
        assert len(labels_batch) == len(label_names)
        labels_batch = {
            label_name: label_batch
                for label_name, label_batch
                    in zip(label_names, labels_batch)
        }
    else:
        assert len(label_names) == 1
        labels_batch = {label_names[0]: labels_batch}

    return labels_batch
