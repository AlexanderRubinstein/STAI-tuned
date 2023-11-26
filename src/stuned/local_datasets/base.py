import numpy as np
import sys
import os
import torch


# local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utility.utils import (
    compute_proportion,
    deterministic_subset,
    load_from_pickle
)
sys.path.pop(0)


class BaseData:

    def __init__(
        self,
        possible_features,
        features_sizes,
        train_val_split,
        file_with_pruned_indices_path=None
    ):
        self.possible_features = possible_features
        self.feature_to_idx = {}
        for feature_id, feature in enumerate(self.possible_features):
            self.feature_to_idx[feature] = feature_id

        # Explanation:
        #
        #   features_sizes = np.array(
        #       [len(values) for values in possible_features_values.values()]
        #   )
        self.features_sizes = features_sizes

        if file_with_pruned_indices_path is None:
            self.pruned_indices = set()
        else:
            self.pruned_indices = load_from_pickle(file_with_pruned_indices_path)
            assert isinstance(self.pruned_indices, set)

        self.train_val_split = train_val_split
        self.train_indices = self._split_into_train_indices()
        self.total_num_samples = self._compute_total_num_samples()

        self.indices_for_feature_values = create_empty_indices_for_feature_values(
            self.possible_features,
            self.features_sizes,
            self.feature_to_idx
        )
        self._fill_indices_for_feature_values()

    def assert_features_and_num_classes(self, features_list, num_classes):
        num_requested_features = len(features_list)
        if num_requested_features == 0:
            raise Exception(
                "Need at least one feature in features list, but it is empty"
            )
        if num_requested_features > len(set(features_list)):
            raise Exception(
                "There are duplicates in features list: {}".format(
                    features_list
                )
            )
        if num_classes < 2:
            raise Exception(
                "Need at least two classes, "
                "while classes requested: {}".format(
                    num_classes
                )
            )
        for feature in features_list:
            self._assert_dataset_specific_feature(feature)
            if feature not in self.possible_features:
                raise Exception(
                    "Feature \"{}\" is not "
                    "from set of possible features: {}".format(
                        feature,
                        self.possible_features
                    )
                )
            num_possible_values \
                = self.features_sizes[self.feature_to_idx[feature]]
            if num_possible_values < num_classes:
                raise Exception(
                    "Number of requested classes is {} "
                    "but number of possible values for feature \"{}\" "
                    "is only {}.".format(
                        num_classes,
                        feature,
                        num_possible_values
                    )
                )

    def _assert_dataset_specific_feature(self, feature):
        pass

    def get_dataset(self):
        raise NotImplementedError()

    def _index_to_features_combination(self, index):
        raise NotImplementedError()

    def _split_into_train_indices(self):

        all_indices = set(list(range(self.total_num_images)))

        non_pruned_set = all_indices.difference(self.pruned_indices)

        return deterministic_subset(
            non_pruned_set,
            compute_proportion(self.train_val_split, len(non_pruned_set))
        )

    def _fill_indices_for_feature_values(self):
        raise NotImplementedError()

    def _compute_total_num_samples(self):
        assert self.train_indices
        total_num_samples = {}
        total_num_samples["train"] = len(self.train_indices)
        total_num_samples["test"] \
            = self.total_num_images \
                - total_num_samples["train"] \
                - len(self.pruned_indices)
        return total_num_samples


def make_index_and_label_pairs_for_single_label(single_label_dataset):
    assert hasattr(single_label_dataset, "indices_for_classes")
    index_and_label_pairs = []
    for class_label in range(len(single_label_dataset.indices_for_classes)):

        indices_for_class_label = list(
            single_label_dataset.indices_for_classes[class_label]
        )

        for idx in indices_for_class_label:
            index_and_label_pairs.append((idx, class_label))

    return sorted(index_and_label_pairs, key=lambda x: x[0])


class ContainerDataset:
    def __init__(
        self,
        images,
        transform=None
    ):
        self.container = images
        self.transform = transform

        self.index_and_label_pairs = self._make_index_and_label_pairs()

    def _make_index_and_label_pairs(self):
        raise NotImplementedError()

    def _extract_images_from_container(self, idx_to_extract):
        raise NotImplementedError()

    def __getitem__(self, idx):

        idx_to_extract, class_label = self.index_and_label_pairs[idx]

        image = self._extract_images_from_container(idx_to_extract)

        image = torch.Tensor(image)
        if self.transform:
            image = self.transform(image)

        if isinstance(class_label, tuple):
            return image, *class_label

        return image, class_label

    def __len__(self):
        return len(self.index_and_label_pairs)


def create_empty_indices_for_feature_values(
        possible_features,
        features_sizes,
        feature_to_idx
    ):
        indices_for_feature_values = {}
        indices_for_feature_values["train"] = {}
        indices_for_feature_values["test"] = {}
        for feature in possible_features:
            num_values = features_sizes[feature_to_idx[feature]]
            indices_for_feature_values["train"][feature] \
                = [
                    set()
                        for _
                        in range(num_values)
                ]
            indices_for_feature_values["test"][feature] = \
                [
                    set()
                        for _
                        in range(num_values)
                ]
        return indices_for_feature_values


def process_feature_classes_combination(
    image_index,
    features_classes_combination,
    train_indices,
    pruned_indices,
    indices_for_feature_values,
    possible_features
):
    if image_index in pruned_indices:
        return

    split_name = (
        "train"
            if image_index in train_indices
                else "test"
    )

    for feature_idx, feature_value_idx in enumerate(
        features_classes_combination
    ):
        (indices_for_feature_values
            [split_name]
            [possible_features[feature_idx]]
            [feature_value_idx]
                .add(image_index)
        )
