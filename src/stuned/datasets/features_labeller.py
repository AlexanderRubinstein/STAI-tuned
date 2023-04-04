import itertools
import copy
import torch
import pickle


# local modules
from ..utility.logger import make_logger
from ..utility.utils import (
    NAME_SEP,
    raise_unknown,
    log_or_print,
    range_for_each_group,
    deterministic_subset,
    compute_proportion
)
from ..datasets.dsprites import (
    make_dsprites_base_data,
    load_dsprites_base_data
)
from ..datasets.utils import (
    make_or_load_from_cache
)
from ..datasets.base import BaseData


SINGLE_LABEL_FOR_OFF_DIAG = False
OFF_DIAG_COMMON_PREFIX = "off-diag"
DIAG_COMMON_PREFIX = "diag"


class FeaturesLabeller:

    def __init__(
        self,
        base_data: BaseData,
        features_list,
        classes_per_feature,
        num_samples_per_cell=None,
        logger=make_logger()
    ):
        '''
        input:
            <base_data> - object of type BaseData
            <features_list> - list of features to use from <base_data>
            <classes_per_feature> - number of classes features' values
                should be uniformly distributed into
            <num_samples_per_cell> - how many images will belong to the
                same class according to all labeling schemes based on features
                from <features_list>

        defines hash function:

            self.indices_for_classes[<split>][<feature>][<label>] -> <indices>
                where <indices> point to images from <split> train/val split
                of <base_data> belonging to class <label>
                according to labeling scheme based on <feature>
        '''

        def get_num_samples_per_cell(total_num_samples_per_cell):
            if total_num_samples_per_cell is not None:
                num_train_samples_per_cell = compute_proportion(
                    self.base_data.train_val_split,
                    total_num_samples_per_cell
                )
                num_test_samples_per_cell \
                    = total_num_samples_per_cell - num_train_samples_per_cell
                assert num_test_samples_per_cell > 0
            else:
                num_train_samples_per_cell = None
                num_test_samples_per_cell = None
            return {
                "train": num_train_samples_per_cell,
                "test": num_test_samples_per_cell
            }

        self.logger = logger

        self.base_data = base_data
        self.features_list = features_list
        self.classes_per_feature = classes_per_feature

        self.diag_name = make_diag_dataset_name(self.features_list)
        self.off_diag_prefix = OFF_DIAG_COMMON_PREFIX
        self.off_diag_names = make_offdiag_dataset_names(
            self.features_list, self.off_diag_prefix
        )

        base_data.assert_features_and_num_classes(
            self.features_list,
            self.classes_per_feature
        )

        self.indices_for_classes = {}
        self.off_diag_indices_to_labels = {}
        self.total_num_samples = {}
        self.num_samples_per_cell = get_num_samples_per_cell(
            num_samples_per_cell
        )
        for split in ["train", "test"]:
            (
                self.indices_for_classes[split],
                self.off_diag_indices_to_labels[split],
                self.total_num_samples[split]
            ) \
                = self._init_for_split(
                    self.base_data,
                    split,
                    self.num_samples_per_cell
                )

    def _init_for_split(self, base_data, split, num_samples_per_cell):


        indices_for_classes_for_split \
            = self._build_indices_for_classes(
                base_data.indices_for_feature_values[split]
            )

        num_samples_per_cell_for_split = num_samples_per_cell[split]
        if num_samples_per_cell_for_split is not None:

            indices_for_classes_for_split, total_num_samples_for_split \
                    = self._restrict_cell_size(
                    indices_for_classes_for_split,
                    num_samples_per_cell_for_split
                )

        else:

            total_num_samples_for_split \
                = self.base_data.total_num_samples[split]

        log_or_print(
            "Using {} samples in \"{}\" split for features labeller.".format(
                total_num_samples_for_split,
                split
            ),
            logger=self.logger
        )

        off_diag_indices_to_labels_for_split \
            = self._build_off_diag_indices_to_labels(
                indices_for_classes_for_split
            )

        return (
            indices_for_classes_for_split,
            off_diag_indices_to_labels_for_split,
            total_num_samples_for_split
        )

    def _build_indices_for_classes(
        self,
        indices_for_feature_values_for_split
    ):

        def update_indices_within_classes(
            indices_to_update,
            indices_to_use_for_update,
            operation
        ):
            assert len(indices_to_update) == len(indices_to_use_for_update)
            for class_label in range(len(indices_to_update)):
                to_update = indices_to_update[class_label]
                to_use_for_update = indices_to_use_for_update[class_label]
                if operation == "intersection":
                    result = to_update.intersection(to_use_for_update)
                elif operation == "difference":
                    result = to_update.difference(to_use_for_update)
                else:
                    raise_unknown(
                        "operation",
                        operation,
                        "update_indices_within_classes"
                    )
                indices_to_update[class_label] = result
            return indices_to_update


        assert self.features_list
        assert self.off_diag_names
        assert self.classes_per_feature

        result = {}
        intersection = None

        for i, feature in enumerate(self.features_list):

            off_diag_name = self.off_diag_names[i]
            assert feature in off_diag_name

            indices_for_classes_for_split_per_dataset \
                = self._merge_values_in_classes(
                    indices_for_feature_values_for_split[feature]
                )

            result[off_diag_name] = indices_for_classes_for_split_per_dataset

            if intersection is None:

                intersection = copy.deepcopy(
                    result[off_diag_name]
                )

            else:

                intersection = update_indices_within_classes(
                    intersection,
                    result[off_diag_name],
                    operation="intersection"
                )

        result[self.diag_name] = intersection

        # remove diag from off-diag
        for off_diag_name in self.off_diag_names:

            result[off_diag_name] \
                = update_indices_within_classes(
                    result[off_diag_name],
                    result[self.diag_name],
                    operation="difference"
                )

        return result

    def _restrict_cell_size(
        self,
        indices_for_classes_for_split,
        requested_cell_size
    ):

        def intersect_with_active_indices(indices_for_classes, active_indices):
            for dataset_name, dataset_indices_for_classes \
                in indices_for_classes.items():

                for class_label, dataset_indices_for_class \
                    in enumerate(dataset_indices_for_classes):

                    indices_for_classes[dataset_name][class_label] \
                        = dataset_indices_for_class.intersection(active_indices)

            return indices_for_classes

        active_indices = self._compute_active_indices(
            indices_for_classes_for_split,
            requested_cell_size
        )

        return (
            intersect_with_active_indices(
                indices_for_classes_for_split,
                active_indices
            ),
            len(active_indices)
        )

    def _compute_active_indices(
        self,
        indices_for_classes_for_split,
        requested_cell_size
    ):
        active_indices = set()
        num_classes = len(
            next(iter(indices_for_classes_for_split.values()))
        )
        assert num_classes

        for labels_combination in itertools.product(
            *[range(num_classes) for _ in range(len(self.features_list))]
        ):

            cell_indices = self._get_cell_indices(
                indices_for_classes_for_split,
                labels_combination
            )
            available_cell_size = len(cell_indices)
            if available_cell_size < requested_cell_size:
                raise Exception(
                    "Requested {} samples per cell,"
                    " but have only {}".format(
                        requested_cell_size,
                        available_cell_size
                    )
                )
            active_indices = active_indices.union(
                deterministic_subset(
                    cell_indices,
                    requested_cell_size
                )
            )

        return active_indices

    def _get_cell_indices(
        self,
        indices_for_classes_for_split,
        labels_combination
    ):

        assert len(self.features_list) == len(labels_combination)
        # diag cell
        if len(set(labels_combination)) == 1:
            return (
                indices_for_classes_for_split
                    [self.diag_name]
                    [labels_combination[0]]
            )
        # off-diag cell
        else:
            return set.intersection(
                *[
                    indices_for_classes_for_split[off_diag_name][label]
                        for off_diag_name, label
                            in zip(self.off_diag_names, labels_combination)
                ]
            )

    def _merge_values_in_classes(self, indices_for_values):

        def compute_num_values_per_class(num_classes, total_values):
            assert total_values >= num_classes
            return [
                group_range[1] - group_range[0]
                    for group_range
                    in range_for_each_group(num_classes, total_values)
            ]

        values_per_class = compute_num_values_per_class(
            self.classes_per_feature,
            len(indices_for_values)
        )

        merged_indices = []
        indices_to_merge = []

        class_idx = 0
        for indices_for_value in indices_for_values:
            indices_to_merge.append(indices_for_value)
            if len(indices_to_merge) == values_per_class[class_idx]:
                merged_indices.append(set.union(*indices_to_merge))
                indices_to_merge = []
                class_idx += 1

        return merged_indices

    def _build_off_diag_indices_to_labels(
        self,
        indices_for_classes_for_split
    ):

        off_diag_indices_to_labels_dict = {}
        for dataset_name, indices_for_classes_for_split_per_dataset \
            in indices_for_classes_for_split.items():

            if dataset_name == self.diag_name:
                continue

            for label, set_of_indices_for_label in enumerate(
                indices_for_classes_for_split_per_dataset
            ):
                for index in set_of_indices_for_label:

                    if index in off_diag_indices_to_labels_dict:
                        off_diag_indices_to_labels_dict[index][dataset_name] \
                            = label
                    else:
                        off_diag_indices_to_labels_dict[index] \
                            = {dataset_name: label}

        return sorted(
            [
                (
                    index,
                    tuple(
                        labels_dict[off_diag_name]
                            for off_diag_name
                                in self.off_diag_names
                    )
                )
                    for index, labels_dict
                        in off_diag_indices_to_labels_dict.items()
            ],
            key=lambda x: x[0]
        )

    def get_dataloaders(
        self,
        train_batch_size,
        test_batch_size,
        num_workers=1,
        single_label=SINGLE_LABEL_FOR_OFF_DIAG
    ):

        train_loaders = {}
        test_loaders = {}

        train_loaders[self.diag_name], test_loaders[self.diag_name] \
            = self._get_dataloaders_for_dataset_name(
                self.diag_name,
                train_batch_size,
                test_batch_size,
                num_workers
            )

        if len(self.features_list) > 1:

            if single_label:

                for off_diag_name in self.off_diag_names:
                    train_loaders[off_diag_name], test_loaders[off_diag_name] \
                        = self._get_dataloaders_for_dataset_name(
                            off_diag_name,
                            train_batch_size,
                            test_batch_size,
                            num_workers
                        )
            else:

                (
                    train_loaders[self.off_diag_prefix],
                    test_loaders[self.off_diag_prefix]
                ) \
                    = self._get_multilabelled_dataloaders_for_off_diag(
                        train_batch_size,
                        test_batch_size,
                        num_workers
                    )

        return train_loaders, test_loaders

    def _get_dataloaders_for_dataset_name(
        self,
        dataset_name,
        train_batch_size,
        test_batch_size,
        num_workers
    ):
        train_loader = self._get_dataloader_from_indices_for_classes(
            self.indices_for_classes["train"][dataset_name],
            train_batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        test_loader = self._get_dataloader_from_indices_for_classes(
            self.indices_for_classes["test"][dataset_name],
            test_batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        return train_loader, test_loader

    def _get_dataloader_from_indices_for_classes(
        self,
        indices_for_classes_for_split_per_dataset,
        batch_size,
        shuffle,
        num_workers
    ):

        return torch.utils.data.DataLoader(
            self.base_data.get_dataset(
                indices_for_classes_for_split_per_dataset
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    def _get_multilabelled_dataloaders_for_off_diag(
        self,
        train_batch_size,
        test_batch_size,
        num_workers
    ):
        train_loader = self._get_multilabelled_off_diag_dataloader(
            self.off_diag_indices_to_labels["train"],
            train_batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        test_loader = self._get_multilabelled_off_diag_dataloader(
            self.off_diag_indices_to_labels["test"],
            test_batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, test_loader

    def _get_multilabelled_off_diag_dataloader(
        self,
        off_diag_indices_to_labels_for_split,
        batch_size,
        shuffle,
        num_workers
    ):
        assert self.off_diag_indices_to_labels
        dataloader = torch.utils.data.DataLoader(
            self.base_data.get_multilabelled_offdiag_dataset(
                off_diag_indices_to_labels_for_split
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        dataloader.label_names = self.off_diag_names
        return dataloader

    def _prepare_for_pickling(self):
        self.logger = None


def make_diag_dataset_name(features_list):
    return "{}{}{}".format(
        DIAG_COMMON_PREFIX,
        NAME_SEP,
        "+".join(features_list)
    )


def make_offdiag_dataset_names(
    features_list,
    off_diag_prefix=OFF_DIAG_COMMON_PREFIX
):
    return [
        "{}{}{}".format(off_diag_prefix, NAME_SEP, feature)
            for feature in features_list
    ]


def make_features_labeller(
    features_labeller_config,
    cache_path,
    logger=make_logger()
):
    base_data_config = features_labeller_config["base_data"]
    base_data_type = base_data_config["type"]
    log_or_print(
        "Making features_labeller for \"{}\"..".format(base_data_type),
        logger=logger
    )
    if base_data_type == "dsprites":
        base_data = make_or_load_from_cache(
            "base_data_dsprites",
            base_data_config[base_data_type],
            make_dsprites_base_data,
            load_dsprites_base_data,
            cache_path=cache_path,
            forward_cache_path=False,
            logger=logger
        )
    else:
        raise_unknown(
            "base data name",
            base_data_type,
            "make_features_labeller"
        )
    return FeaturesLabeller(
        base_data,
        features_labeller_config["features_list"],
        features_labeller_config["num_classes_per_feature"],
        features_labeller_config["num_samples_per_cell"],
        logger=logger
    )


def load_features_labeller(path):
    return pickle.load(open(path, "rb"))
