# local modules
from ..utility.logger import make_logger
from ..utility.utils import raise_unknown
from .features_labeller import (
    make_features_labeller,
    load_features_labeller
)
from .tiny_imagenet import (
    get_tiny_imagenet_dataloaders,
    get_tiny_imagenet_test_projector_dataset
)
from .dsprites import (
    get_dsprites_unlabeled_data
)
from .utils import (
    randomly_subsampled_dataloader,
    make_or_load_from_cache
)


TRAIN_DATA_PERCENT_FOR_EVAL = 0.1
EVAL_ON_TRAIN_LOGS_NAME = (
    "random ({})-fraction"
    " of train data with frozen weights").format(
        TRAIN_DATA_PERCENT_FOR_EVAL
    )


def get_dataloaders(experiment_config, logger=make_logger()):

    data_config = experiment_config["data"]
    params_config = experiment_config["params"]
    dataset_config = data_config["dataset"]
    dataset_type = dataset_config["type"]
    cache_path = experiment_config["cache_path"]

    eval_batch_size = params_config["eval"]["batch_size"]
    train_batch_size = params_config["train"]["batch_size"]

    if dataset_type == "features_labeller":

        features_labeller_config = dataset_config[dataset_type]
        features_labeller = make_or_load_from_cache(
            "features_labeller_with_{}".format(
                features_labeller_config["base_data"]["type"]
            ),
            features_labeller_config,
            make_features_labeller,
            load_features_labeller,
            cache_path=cache_path,
            forward_cache_path=True,
            logger=logger
        )
        trainloaders, testloaders = features_labeller.get_dataloaders(
            train_batch_size,
            eval_batch_size,
            data_config["num_data_readers"]
        )
        trainloader = trainloaders[features_labeller.diag_name]

    elif dataset_type == "tiny-imagenet":
        trainloader, testloaders \
            = get_tiny_imagenet_dataloaders(
                tiny_imagenet_config=dataset_config[dataset_type],
                params_config=params_config,
                logger=logger
            )

    else:
        raise_unknown(
            "dataset type",
            dataset_type,
            "data config"
        )

    # add train subset into test dataloaders
    testloaders[EVAL_ON_TRAIN_LOGS_NAME] \
        = randomly_subsampled_dataloader(
            trainloader,
            max(
                1,
                round(TRAIN_DATA_PERCENT_FOR_EVAL * len(trainloader.dataset))
            ),
            batch_size=eval_batch_size
        )

    return trainloader, testloaders


def get_manifold_projector_dataset(
    projector_data_config,
    cache_path,
    logger=make_logger()
):
    manifold_projector_dataset_type = projector_data_config["type"]

    n_images = projector_data_config["total_number_of_images"]
    assert n_images >= 0

    projector_dataset_config \
        = projector_data_config[manifold_projector_dataset_type]

    if manifold_projector_dataset_type == "tiny-imagenet-test":

        return get_tiny_imagenet_test_projector_dataset(
            manifold_projector_dataset_type,
            projector_dataset_config,
            n_images,
            logger
        )

    elif manifold_projector_dataset_type == "dsprites":

        return get_dsprites_unlabeled_data(
            projector_dataset_config,
            "train",
            n_images,
            cache_path,
            logger
        )

    else:
        raise_unknown(
            "data for manifold projector",
            manifold_projector_dataset_type,
            "manifold projector data config"
        )
