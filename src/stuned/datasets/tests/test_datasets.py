import sys
import os
import torch
import PIL.Image
import copy
import numpy as np
import itertools


# local modules
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
from stuned.utility.helpers_for_tests import (
    wrapper_for_test,
    wrapper_for_test_with_extended_args,
    expect_failed
)
from stuned.utility.utils import (
    get_stuned_root_path,
    update_dict_by_nested_key,
    error_callback,
    coefficients_for_bases,
    compute_proportion,
    assert_two_values_are_close
)
from stuned.datasets.common import (
    get_dataloaders,
    EVAL_ON_TRAIN_LOGS_NAME
)
from stuned.datasets.utils import (
    DEFAULT_CACHE_PATH,
    make_or_load_from_cache
)
from stuned.datasets.features_labeller import (
    make_dsprites_base_data,
    make_features_labeller,
    load_features_labeller
)
from stuned.datasets.dsprites import (
    load_dsprites_base_data,
    get_dsprites_images_as_tensor_generator
)


IMAGE_SCALING_FACTOR = 255
DICT_KEY_SEPARATOR = "___"
RANDOM_SEED = 14
BATCH_SIZE = 16
N_IMAGES_FOR_TENSOR = 100


MINIMAL_EXP_CONFIG = {
    'cache_path': DEFAULT_CACHE_PATH,
    'params': {
        'train': {
            'batch_size': 16
        },
        'eval': {
            'batch_size': 1
        }
    }
}
MINIMAL_DATA_CONFIG = {
    'num_data_readers': 1,
    "dataset": {}
}
DSPRITES_CONFIG = {
    'type': 'dsprites',
    'dsprites': {
        'type': "bw",
        'train_val_split': 0.8,
        'path': os.path.join(get_stuned_root_path(), "data", "dsprites"),
    }
}
FEATURES_LABELLER_CONFIG = {
    'type': 'features_labeller',
    'features_labeller': {
        'features_list': ['shape', 'scale', 'orientation'],
        'num_classes_per_feature': 3,
        'num_samples_per_cell': 10,
        'base_data': copy.deepcopy(DSPRITES_CONFIG)
    }
}
TINY_IMAGENET_CONFIG = {
    'type': 'tiny-imagenet',
    'tiny-imagenet': {
        'total_number_of_samples': 100,
        'train_val_split': 0.8,
        'normalize': True,
        'path': os.path.join(get_stuned_root_path(), "data", "tiny-imagenet"),
    }
}
TENSOR_GENERATOR_BATCH_SIZE = 3


def get_dataloader_len(dataloader):
    cnt = 0
    for input, _ in dataloader:
        cnt += input.shape[0]
    return cnt


@wrapper_for_test
def test_reproducible_dataloaders():

    def test_for_config(specific_data_config):
        test_reproducibility_for_config(
            get_config_with_data(specific_data_config)
        )

    def test_reproducibility_for_config(experiment_config):
        _, testloaders_one = get_dataloaders(experiment_config)
        _, testloaders_two = get_dataloaders(experiment_config)
        testloaders_one.pop(EVAL_ON_TRAIN_LOGS_NAME)
        testloaders_two.pop(EVAL_ON_TRAIN_LOGS_NAME)
        assert_same_dataloader_dicts(testloaders_one, testloaders_two)

    features_labeller_config = copy.deepcopy(FEATURES_LABELLER_CONFIG)
    test_for_config(features_labeller_config)
    update_dict_by_nested_key(
        features_labeller_config,
        [
            "features_labeller",
            "base_data",
            "dsprites",
            "type"
        ],
        "color"
    )
    test_for_config(features_labeller_config)
    test_for_config(TINY_IMAGENET_CONFIG)


def get_config_with_data(specific_dataset_config):
    experiment_config = copy.deepcopy(MINIMAL_EXP_CONFIG)
    data_config = copy.deepcopy(MINIMAL_DATA_CONFIG)
    data_config["dataset"] = copy.deepcopy(specific_dataset_config)
    experiment_config["data"] = data_config
    return experiment_config


def assert_same_dataloader_dicts(dataloaders_dict_one, dataloaders_dict_two):
    assert set(dataloaders_dict_one.keys()) == set(dataloaders_dict_two.keys())
    for test_dataloader_key in dataloaders_dict_one.keys():
        assert_same_dataloaders(
            dataloaders_dict_one[test_dataloader_key],
            dataloaders_dict_two[test_dataloader_key]
        )


def assert_same_dataloaders(dataloader_one, dataloader_two):

    for dataloader_items_one, dataloader_items_two in zip(
        dataloader_one,
        dataloader_two
    ):
        for item_one, item_two in zip(
            dataloader_items_one,
            dataloader_items_two
        ):

            assert torch.isclose(item_one, item_two).all()


@wrapper_for_test_with_extended_args
def test_dsprites_base_data(extended_args):

    def get_image_by_features_classes_combination(
        dsprites_base_data,
        features_classes_combination
    ):
        image_idx = dsprites_base_data._features_classes_combination_to_index(
            features_classes_combination
        )
        dataset = dsprites_base_data.get_dataset([set([image_idx])])
        image, _ = dataset[0]
        image = image.numpy()
        if len(image.shape) == 3:
            image = image.transpose(1, 2, 0)

        return (image * IMAGE_SCALING_FACTOR).astype(np.uint8)

    def test_image_for_classes_combination(
        base_data_dsprites,
        expected_image_path,
        features_classes_combination
    ):
        image = get_image_by_features_classes_combination(
            base_data_dsprites,
            features_classes_combination
        )
        convert_to = "RGB" if len(image.shape) == 3 else "L"

        expected_image = np.array(PIL.Image.open(
            expected_image_path
        ).convert(convert_to))

        assert np.isclose(expected_image, image).all()


    def test_indices_to_values(base_data_dsprites, pool, shared_memory_manager):

        indices_for_feature_values \
            = base_data_dsprites.indices_for_feature_values

        feature_bases = base_data_dsprites.features_bases

        train_val_split = base_data_dsprites.train_val_split

        total_samples = base_data_dsprites.imgs.shape[0]

        if base_data_dsprites.colored:
            total_samples *= 3

        num_train_samples = compute_proportion(train_val_split, total_samples)

        num_test_samples = total_samples - num_train_samples

        expected_samples_in_split = {
            "train": num_train_samples,
            "test": num_test_samples
        }

        total_samples_in_split = shared_memory_manager.dict()
        lock = shared_memory_manager.Lock()

        for split in ["train", "test"]:
            for feature, indices_per_feature \
                in indices_for_feature_values[split].items():

                feature_id = base_data_dsprites.feature_to_idx[feature]
                starmap_args = [
                    (
                        indices,
                        expected_value_id,
                        feature_id,
                        split,
                        total_samples_in_split,
                        lock,
                        feature_bases
                    )
                        for expected_value_id, indices
                            in enumerate(indices_per_feature)
                ]
                pool.starmap_async(
                    pool_test_indices_to_values,
                    starmap_args,
                    error_callback=error_callback
                )

        pool.close()
        pool.join()

        for shared_dict_key, num_samples in total_samples_in_split.items():
            split = shared_dict_key.split(DICT_KEY_SEPARATOR)[0]
            assert num_samples == expected_samples_in_split[split]

    dsprites_config = copy.deepcopy(DSPRITES_CONFIG["dsprites"])

    test_data_folder = os.path.join(
        os.path.dirname(__file__),
        "data"
    )
    base_data_dsprites_bw = make_or_load_from_cache(
        "base_data_dsprites",
        dsprites_config,
        make_dsprites_base_data,
        load_dsprites_base_data,
        cache_path=DEFAULT_CACHE_PATH,
        forward_cache_path=False,
        logger=extended_args["logger"]
    )
    test_image_for_classes_combination(
        base_data_dsprites_bw,
        os.path.join(test_data_folder, "grey_example.png"),
        np.array([0, 2, 4, 20, 16, 16])
    )
    dsprites_config["type"] = "color"
    base_data_dsprites_color = make_or_load_from_cache(
        "base_data_dsprites",
        dsprites_config,
        make_dsprites_base_data,
        load_dsprites_base_data,
        cache_path=DEFAULT_CACHE_PATH,
        forward_cache_path=False,
        logger=extended_args["logger"]
    )
    test_image_for_classes_combination(
        base_data_dsprites_color,
        os.path.join(test_data_folder, "color_example1.png"),
        np.array([0, 0, 0, 0, 0, 0])
    )
    test_image_for_classes_combination(
        base_data_dsprites_color,
        os.path.join(test_data_folder, "color_example2.png"),
        np.array([1, 1, 1, 1, 1, 1])
    )
    test_image_for_classes_combination(
        base_data_dsprites_color,
        os.path.join(test_data_folder, "color_example3.png"),
        np.array([2, 2, 5, 39, 31, 31])
    )

    test_indices_to_values(
        base_data_dsprites_color,
        pool=extended_args["pool"],
        shared_memory_manager=extended_args["shared_memory_manager"]
    )


@wrapper_for_test_with_extended_args
def test_dsprites_feature_labeller(extended_args):

    def test_for_feature_labeller_config(features_labeller_config):

        def test_multilabel_dataloader(
            features_labeller,
            eval_batch_size,
            num_data_readers
        ):

            def compare_single_and_multilabel_dataloaders(
                singlelabel_dataloaders,
                multilabel_dataloader,
                max_samples_to_check=1000
            ):
                label_names = multilabel_dataloader.label_names
                assert sorted(label_names) \
                    == sorted(list(singlelabel_dataloaders.keys()))
                singlelabel_dataloaders_list = [
                    singlelabel_dataloaders[singlelabel_dataloader_name]
                        for singlelabel_dataloader_name in label_names
                ]
                for i, dataloaders_items in enumerate(zip(
                    multilabel_dataloader,
                    *singlelabel_dataloaders_list
                )):
                    if i == max_samples_to_check:
                        break

                    assert (
                        len(label_names) + 1 == len(dataloaders_items)
                    )
                    multilabel_dataloader_item = dataloaders_items[0]
                    assert (
                        len(multilabel_dataloader_item) == len(label_names) + 1
                    )
                    image = multilabel_dataloader_item[0]
                    multiple_labels = multilabel_dataloader_item[1:]

                    for label_pos, singlelabel_dataloader_items in enumerate(
                        dataloaders_items[1:]
                    ):
                        singlelabel_image, single_label \
                            = singlelabel_dataloader_items
                        assert torch.isclose(singlelabel_image, image).all()
                        assert torch.isclose(
                            single_label,
                            multiple_labels[label_pos]
                        ).all()

            _, singlelabel_test_loaders = features_labeller.get_dataloaders(
                eval_batch_size,
                eval_batch_size,
                num_data_readers,
                single_label=True
            )


            _, multilabel_test_loaders = features_labeller.get_dataloaders(
                eval_batch_size,
                eval_batch_size,
                num_data_readers,
                single_label=False
            )

            singlelabel_test_loaders.pop(features_labeller.diag_name)
            multilabel_test_loader \
                = multilabel_test_loaders[features_labeller.off_diag_prefix]

            compare_single_and_multilabel_dataloaders(
                singlelabel_test_loaders,
                multilabel_test_loader
            )

        exp_config = copy.deepcopy(MINIMAL_EXP_CONFIG)

        exp_config["data"] = copy.deepcopy(MINIMAL_DATA_CONFIG)

        features_labeller = make_or_load_from_cache(
            "features_labeller_with_dsprites",
            features_labeller_config,
            make_features_labeller,
            load_features_labeller,
            cache_path=DEFAULT_CACHE_PATH,
            forward_cache_path=True,
            logger=extended_args["logger"]
        )

        data_config = exp_config["data"]
        params_config = exp_config["params"]

        train_batch_size = params_config["train"]["batch_size"]
        eval_batch_size = params_config["eval"]["batch_size"]
        num_data_readers = data_config["num_data_readers"]

        test_multilabel_dataloader(
            features_labeller,
            eval_batch_size,
            num_data_readers
        )

        train_loaders, test_loaders = features_labeller.get_dataloaders(
            train_batch_size,
            eval_batch_size,
            num_data_readers
        )

        num_classes = features_labeller_config["num_classes_per_feature"]
        num_samples_per_cell = features_labeller_config["num_samples_per_cell"]
        features_list = features_labeller_config["features_list"]

        train_val_split = features_labeller.base_data.train_val_split
        num_cells = num_classes ** len(features_list)
        expected_num_samples_per_train_cell = int(
            features_labeller.total_num_samples["train"] / num_cells
        )
        expected_num_samples_per_test_cell = int(
            features_labeller.total_num_samples["test"] / num_cells
        )
        if num_samples_per_cell is not None:
            assert expected_num_samples_per_train_cell \
                == compute_proportion(train_val_split, num_samples_per_cell)
            assert expected_num_samples_per_test_cell \
                == num_samples_per_cell - expected_num_samples_per_train_cell

        test_dataset_size(
            train_loaders,
            test_loaders,
            features_labeller.diag_name,
            expected_num_samples_per_train_cell,
            expected_num_samples_per_test_cell,
            num_cells,
            num_classes,
            (num_samples_per_cell is None)
        )

        test_indices_for_classes(
            features_labeller,
            num_classes,
            expected_num_samples_per_train_cell,
            expected_num_samples_per_test_cell,
            (num_samples_per_cell is None)
        )

    def test_indices_for_classes(
        features_labeller,
        num_classes,
        expected_num_samples_per_train_cell,
        expected_num_samples_per_test_cell,
        use_all_samples
    ):

        def assert_sizes(
            num_train_indices_per_class,
            num_test_indices_per_class,
            expected_num_train_samples_per_class,
            expected_num_test_samples_per_class,
            use_all_samples
        ):

            # due to discrete nature of feature values,
            # samples with the same feature value
            # can be distributed between train/val with proportion
            # slightly different from <train_val_split>,
            # therefore without fixing cell size their split-wise number
            # can not be computed
            # as number samples in split divided by number of cells
            if use_all_samples:
                assert (
                    (
                        num_train_indices_per_class
                            + num_test_indices_per_class
                    )
                        == (
                            expected_num_train_samples_per_class
                                + expected_num_test_samples_per_class
                        )
                    )
            else:
                assert num_train_indices_per_class \
                    == expected_num_train_samples_per_class
                assert num_test_indices_per_class \
                    == expected_num_test_samples_per_class

        def assert_num_classes(features_labeller, num_classes):
            for split in ["train", "test"]:
                for dataset_indices \
                    in features_labeller.indices_for_classes[split].values():
                    assert len(dataset_indices) == num_classes

        def get_samples_per_class(
            expected_num_samples_per_cell,
            total_samples,
            num_classes
        ):

            expected_samples_per_class_for_diag = expected_num_samples_per_cell
            expected_samples_per_class_for_offdiag \
                = int(total_samples / num_classes) \
                    - expected_samples_per_class_for_diag

            return (
                expected_samples_per_class_for_offdiag,
                expected_samples_per_class_for_diag
            )

        def assert_samples_per_cell(
            features_labeller,
            expected_num_samples_per_cell
        ):

            for split in ["train", "test"]:
                for labels_combination in itertools.product(
                    *[
                        range(num_classes)
                            for _
                                in range(len(features_labeller.features_list))
                    ]
                ):

                    cell = features_labeller._get_cell_indices(
                        features_labeller.indices_for_classes[split],
                        labels_combination
                    )
                    assert len(cell) == expected_num_samples_per_cell[split]

        assert_num_classes(features_labeller, num_classes)

        (
            expected_num_train_samples_per_class_for_offdiag,
            expected_num_train_samples_per_class_for_diag
        ) = get_samples_per_class(
            expected_num_samples_per_train_cell,
            features_labeller.total_num_samples["train"],
            num_classes
        )

        (
            expected_num_test_samples_per_class_for_offdiag,
            expected_num_test_samples_per_class_for_diag
        ) = get_samples_per_class(
            expected_num_samples_per_test_cell,
            features_labeller.total_num_samples["test"],
            num_classes
        )

        for (
            (train_dataset_name, train_indices_for_classes_per_dataset),
            (test_dataset_name, test_indices_for_classes_per_dataset)
        ) in zip(
            features_labeller.indices_for_classes["train"].items(),
            features_labeller.indices_for_classes["test"].items()
        ):
            assert train_dataset_name == test_dataset_name
            for train_indices_per_class, test_indices_per_class in zip(
                train_indices_for_classes_per_dataset,
                test_indices_for_classes_per_dataset
            ):
                if train_dataset_name == features_labeller.diag_name:
                    assert_sizes(
                        len(train_indices_per_class),
                        len(test_indices_per_class),
                        expected_num_train_samples_per_class_for_diag,
                        expected_num_test_samples_per_class_for_diag,
                        use_all_samples
                    )
                else:
                    assert_sizes(
                        len(train_indices_per_class),
                        len(test_indices_per_class),
                        expected_num_train_samples_per_class_for_offdiag,
                        expected_num_test_samples_per_class_for_offdiag,
                        use_all_samples
                    )

        if not use_all_samples:
            expected_num_samples_per_cell = {
                "train": expected_num_samples_per_train_cell,
                "test": expected_num_samples_per_test_cell
            }
            assert_samples_per_cell(
                features_labeller,
                expected_num_samples_per_cell
            )

    def test_dataset_size(
        train_dataloaders,
        test_dataloaders,
        diag_name,
        expected_num_samples_per_train_cell,
        expected_num_samples_per_test_cell,
        num_cells,
        num_classes,
        use_all_samples
    ):

        def compute_diag_sizes(
            expected_num_samples_per_cell,
            num_cells,
            num_classes
        ):
            expected_train_diag_size \
                = expected_num_samples_per_cell * num_classes
            return (
                (
                    expected_num_samples_per_cell
                        * num_cells
                        - expected_train_diag_size
                ),
                expected_train_diag_size
            )

        def assert_sizes(
            train_dataloader,
            test_dataloader,
            expected_train_size,
            expected_test_size,
            diag_when_use_all_samples
        ):

            if diag_when_use_all_samples:
                assert (
                    (
                        len(train_dataloader.dataset)
                            + len(test_dataloader.dataset)
                    )
                        == (
                            expected_train_size
                                + expected_test_size
                        )
                )
            else:
                assert len(train_dataloader.dataset) == expected_train_size
                assert len(test_dataloader.dataset) == expected_test_size

        expected_train_offdiag_size, expected_train_diag_size \
            = compute_diag_sizes(
                expected_num_samples_per_train_cell,
                num_cells,
                num_classes
            )

        expected_test_offdiag_size, expected_test_diag_size \
            = compute_diag_sizes(
                expected_num_samples_per_test_cell,
                num_cells,
                num_classes
            )

        for (
                (train_dataset_name, train_dataloader),
                (test_dataset_name, test_dataloader)
            ) in zip(train_dataloaders.items(), test_dataloaders.items()):

            assert train_dataset_name == test_dataset_name
            if train_dataset_name == diag_name:
                assert_sizes(
                    train_dataloader,
                    test_dataloader,
                    expected_train_diag_size,
                    expected_test_diag_size,
                    use_all_samples
                )
            else:
                assert_sizes(
                    train_dataloader,
                    test_dataloader,
                    expected_train_offdiag_size,
                    expected_test_offdiag_size,
                    False
                )

    base_data_config = copy.deepcopy(DSPRITES_CONFIG)

    features_labeller_config = {
        "features_list": ["posX", "posY"],
        "num_classes_per_feature": 32,
        "num_samples_per_cell": None,
        "base_data": base_data_config
    }

    test_for_feature_labeller_config(features_labeller_config)

    fail_info = {"msg": None}
    should_fail = expect_failed(test_for_feature_labeller_config, fail_info)

    # color feature for bw dataset
    features_labeller_config["features_list"] = ["color", "shape"]
    fail_info["msg"] = "Should fail with color for bw"
    should_fail(features_labeller_config)

    base_data_config["dsprites"]["type"] = "color"

    features_labeller_config = {
        "features_list": [
            "posY",
            "shape",
            "color",
            "scale",
            "posX",
            "orientation"
        ],
        "num_classes_per_feature": 3,
        "num_samples_per_cell": 100,
        "base_data": base_data_config
    }

    test_for_feature_labeller_config(features_labeller_config)

    # too big cell size
    features_labeller_config["num_samples_per_cell"] = 100500
    fail_info["msg"] = "Should fail with too big cell size requested"
    should_fail(features_labeller_config)
    features_labeller_config["num_samples_per_cell"] = 100

    # too many classes
    features_labeller_config["num_classes_per_feature"] = 4
    fail_info["msg"] = "Should fail with too many classes requested"
    should_fail(features_labeller_config)

    # too few classes
    features_labeller_config["num_classes_per_feature"] = 1
    fail_info["msg"] = "Should fail with too few classes requested"
    should_fail(features_labeller_config)

    features_labeller_config["num_classes_per_feature"] = 3

    # wrong feature
    features_labeller_config["features_list"] = ["color", "lol"]
    fail_info["msg"] = "Should fail with wrong feature"
    should_fail(features_labeller_config)

    # duplicate features
    features_labeller_config["features_list"] = ["color", "color"]
    fail_info["msg"] = "Should fail with duplicates in features_list"
    should_fail(features_labeller_config)

    # empty features_list
    features_labeller_config["features_list"] = []
    fail_info["msg"] = "Should fail with empty features_list"
    should_fail(features_labeller_config)


def pool_test_indices_to_values(
    indices,
    expected_value_id,
    feature_id,
    split,
    total_samples_in_split,
    lock,
    feature_bases
):

    assert len(indices) > 0
    shared_dict_key = split + DICT_KEY_SEPARATOR + str(feature_id)
    with lock:
        total_samples_in_split[shared_dict_key] \
            = total_samples_in_split.setdefault(shared_dict_key, 0) \
                + len(indices)

    for index in indices:
        features_combination = coefficients_for_bases(index, feature_bases)
        found_value_id = features_combination[feature_id]
        assert expected_value_id == found_value_id


@wrapper_for_test_with_extended_args
def test_convert_dsprites_to_tensor(extended_args):

    def parametrized_test(dsprites_config, image_indices, total_images, logger):

        base_data_dsprites_colored = make_or_load_from_cache(
            "base_data_dsprites",
            dsprites_config,
            make_dsprites_base_data,
            load_dsprites_base_data,
            cache_path=DEFAULT_CACHE_PATH,
            forward_cache_path=False,
            logger=logger
        )
        dataset = base_data_dsprites_colored.get_dataset(
            [set(image_indices)]
        )

        inputs_as_tensor = None

        for inputs_batch_as_tensor in get_dsprites_images_as_tensor_generator(
            dsprites_config,
            total_images,
            TENSOR_GENERATOR_BATCH_SIZE,
            logger,
            cache_path=DEFAULT_CACHE_PATH
        ):

            if inputs_as_tensor is None:
                inputs_as_tensor = inputs_batch_as_tensor
            else:
                inputs_as_tensor = torch.cat(
                    (inputs_as_tensor, inputs_batch_as_tensor)
                )

        for image_idx in image_indices:

            original_image, original_label = dataset[image_idx]
            image_from_tensor = inputs_as_tensor[image_idx]

            assert_two_values_are_close(original_image, image_from_tensor)

    logger = extended_args["logger"]
    dsprites_config = copy.deepcopy(DSPRITES_CONFIG["dsprites"])
    dsprites_config["type"] = "color"
    total_images = N_IMAGES_FOR_TENSOR
    image_indices = list(range(total_images))

    parametrized_test(dsprites_config, image_indices, total_images, logger)


def main():
    test_dsprites_base_data()
    test_dsprites_feature_labeller()
    test_reproducible_dataloaders()
    test_convert_dsprites_to_tensor()


if __name__ == "__main__":
    main()
