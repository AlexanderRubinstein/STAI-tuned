import os
import torch
import numpy as np
import itertools
import random
from torchvision import transforms as T
import pickle
import sys
import os


# local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from local_datasets.utils import (
    EMPTY_URL,
    fetch_data,
    assert_dataset_as_file,
    do_fetch_dataset_as_file,
    convert_dataset_to_tensor_generator,
    make_or_load_from_cache
)  # slow
from local_datasets.base import BaseData
from utility.logger import make_logger
from utility.utils import (
    raise_unknown,
    coefficients_for_bases,
    log_or_print,
    compute_proportion,
    deterministically_subsample_indices_uniformly,
    get_project_root_path
)
from local_datasets.transforms import (
    make_transforms
)
sys.path.pop(0)


DSPRITES_ORIGINAL_URL = "https://disk.yandex.ru/d/v_FfJcdeNKQVNg"
DSPRITES_BW_URL = "https://disk.yandex.ru/d/0BJBFes548cOhg"
# if "dsprites_color.h5" is needed, it should be generated locally,
# using scripts from the "hyp_bias" repo
DSPRITES_COLOR_URL = EMPTY_URL
DSPRITES_ORIGINAL_HASH = "7da33b31b13a06f4b04a70402ce90c2e"
DSPRITES_BW_HASH = "322531291bed2d9f98049bae4e64d194"
DSPRITES_COLOR_HASH = "ce1f3448066b87edada5460376289731"


DSPRITES_NUM_COLORS = 3
MAX_TOTAL_NUMBER_OF_DSPRITES_SAMPLES = 737280


def get_default_dsprites_config():
    return {
        'type': "color",
        'train_val_split': 1.0,
        'path': os.path.join(get_project_root_path(), "data", "dsprites")
    }


def get_dsprites_datasaet_name(
    dsprites_config
):
    return "dsprites_" + str(dsprites_config["type"])


def get_dsprites_dataset_url(dataset_name):
    if dataset_name == "dsprites":
        return DSPRITES_ORIGINAL_URL
    elif dataset_name == "dsprites_bw":
        return DSPRITES_BW_URL
    elif dataset_name == "dsprites_color":
        return DSPRITES_COLOR_URL
    else:
        raise_unknown(
            "dataset name",
            dataset_name,
            "get_dsprites_dataset_url"
        )


def get_dsprites_file_hash(dataset_name):
    if dataset_name == "dsprites":
        return DSPRITES_ORIGINAL_HASH
    elif dataset_name == "dsprites_bw":
        return DSPRITES_BW_HASH
    elif dataset_name == "dsprites_color":
        return DSPRITES_COLOR_HASH
    else:
        raise_unknown(
            "dataset name",
            dataset_name,
            "get_dsprites_file_hash"
        )


def get_dsprites_projector_dataset_v2(dsprites_config, n_images, logger):

    inputs_as_tensor = next(iter(get_dsprites_images_as_tensor_generator(
        dsprites_config,
        n_images,
        n_images,
        logger
    )))
    inputs_as_tensor = torch.flatten(
        inputs_as_tensor,
        start_dim=1
    )
    # dummy labels for back-compatibility
    labels_as_tensor = torch.zeros(inputs_as_tensor.shape[0], 1)
    dataset = torch.utils.data.TensorDataset(
        inputs_as_tensor,
        labels_as_tensor
    )
    return dataset


def get_dsprites_images_as_tensor_generator(
    dsprites_config,
    n_images,
    batch_size,
    logger,
    cache_path=None
):

    # TODO(Alex|23.01.2023) make None logger available
    # for "make_or_load_from_cache"
    assert logger is not None

    dsprites_base_data = make_or_load_from_cache(
        "base_data_dsprites",
        dsprites_config,
        make_dsprites_base_data,
        load_dsprites_base_data,
        cache_path=cache_path,
        forward_cache_path=False,
        logger=logger
    )

    if n_images == 0:
        n_images = MAX_TOTAL_NUMBER_OF_DSPRITES_SAMPLES
        if dsprites_config["type"] == "color":
            n_images *= DSPRITES_NUM_COLORS

    dataset = dsprites_base_data.get_unlabelled_dataset(list(range(n_images)))

    for inputs_batch_as_tensor, _ in convert_dataset_to_tensor_generator(
        dataset,
        batch_size
    ):

        yield inputs_batch_as_tensor


class DSpritesDataset:

    def __init__(
        self,
        dsprites_images,
        colored=False,
        transform=None
    ):
        self.dsprites_images = dsprites_images
        self.transform = transform

        self.colored = colored
        if self.colored:
            self.colored_image_shape \
                = (DSPRITES_NUM_COLORS,) \
                    + dsprites_images[0].shape

        self.num_images_of_one_color = self.dsprites_images.shape[0]
        self.index_and_label_pairs = self._make_index_and_label_pairs()

    def _make_index_and_label_pairs(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        idx_to_extract, class_label = self.index_and_label_pairs[idx]

        if self.colored:
            assert self.colored_image_shape
            color = int(idx_to_extract / self.num_images_of_one_color)
            original_idx = idx_to_extract % self.num_images_of_one_color
            colored_image = np.zeros(self.colored_image_shape)
            colored_image[color, ...] = self.dsprites_images[original_idx]
            image = colored_image
        else:
            image = self.dsprites_images[idx_to_extract]

        image = torch.Tensor(image)
        if self.transform:
            image = self.transform(image)

        if isinstance(class_label, tuple):
            return image, *class_label

        return image, class_label

    def __len__(self):
        return len(self.index_and_label_pairs)


class DSpritesUnlabelledDataset(DSpritesDataset):

    def __init__(
        self,
        dsprites_images,
        indices_to_extract,
        colored=False,
        transform=None
    ):

        self.indices_and_labels_pairs = [(i, 0) for i in indices_to_extract]
        super(DSpritesUnlabelledDataset, self).__init__(
            dsprites_images,
            colored,
            transform
        )

    def _make_index_and_label_pairs(self):

        assert self.indices_and_labels_pairs
        return self.indices_and_labels_pairs


class DSpritesSingleLabelDataset(DSpritesDataset):

    def __init__(
        self,
        dsprites_images,
        indices_for_classes_to_extract,
        colored=False,
        transform=None
    ):
        self.indices_for_classes = indices_for_classes_to_extract
        super(DSpritesSingleLabelDataset, self).__init__(
            dsprites_images,
            colored,
            transform
        )

    def _make_index_and_label_pairs(self):
        assert self.indices_for_classes
        index_and_label_pairs = []
        for class_label in range(len(self.indices_for_classes)):

            indices_for_class_label = list(
                self.indices_for_classes[class_label]
            )

            for idx in indices_for_class_label:
                index_and_label_pairs.append((idx, class_label))

        return sorted(index_and_label_pairs, key=lambda x: x[0])


class DSpritesMultiLabelDataset(DSpritesDataset):

    def __init__(
        self,
        dsprites_images,
        indices_to_labels_to_extract,
        colored=False,
        transform=None
    ):

        self.indices_to_labels = indices_to_labels_to_extract
        super(DSpritesMultiLabelDataset, self).__init__(
            dsprites_images,
            colored,
            transform
        )

    def _make_index_and_label_pairs(self):

        assert self.indices_to_labels
        return self.indices_to_labels


class BaseDataDSprites(BaseData):

    def __init__(self, dsprites_zip, colored, train_val_split, transform=None):

        self.colored = colored
        self.imgs = dsprites_zip['imgs']
        self.total_num_images = self.imgs.shape[0] * (3 if self.colored else 1)
        metadata = dsprites_zip['metadata'][()]

        possible_features = metadata['latents_names']
        possible_features_values = metadata['latents_possible_values']
        if colored:
            possible_features_values['color'] = np.array([1., 2., 3.])

        self.transform = transform

        super(BaseDataDSprites, self).__init__(
            possible_features,
            possible_features_values,
            train_val_split
        )

    def _init_helpers(self):
        self.features_bases = np.concatenate(
            (self.features_sizes[::-1].cumprod()[::-1][1:], np.array([1,]))
        )

    def _compute_total_num_samples(self):
        assert self.train_indices
        total_num_samples = {}
        total_num_samples["train"] = len(self.train_indices)
        total_num_samples["test"] \
            = self.total_num_images \
                - total_num_samples["train"]
        return total_num_samples

    def _split_into_train_indices(self):

        return set(deterministically_subsample_indices_uniformly(
            self.total_num_images,
            compute_proportion(self.train_val_split, self.total_num_images)
        ).numpy().tolist())

    # currently not used
    def _split_into_train_indices_by_combinations_subsampling(
        self,
        frozen_combination_part_length=3
    ):
        '''
        input:
            <frozen_combination_part_length> - how many positions
                in combination will be "frozen"

        output:
            <train_indices> - indices of images in train split

        description:
            considers feature values combination in the following way:
                features_values_combination = [
                    frozen_combination_part,
                    varying_combination_part
                ]
            where "len(frozen_combination_part)"
            equals <frozen_combination_part_length>;
            for each frozen combination part subsample <self.train_val_split>
            varying combination parts;
            construct "features_values_combination" by concatenating
            "frozen_combination_part" and subsampled "varying_combination_part";
            convert constructed combinations into indices
            and put them into "train_indices" set;
                e.g. [0, 1, 2] - current frozen combination
                of values for ['color', 'shape', 'scale'];
                this method subsamples combinations from all combinations
                that look like [0, 1, 2, *, *, *];
                if one of subsampled varying combinations
                of values for ["orientation", "posX", "posY"]
                is [3, 4, 5] then "features_values_combination"
                will be [0, 1, 2, 3, 4, 5] that converts into some index by
                "index = self._features_classes_combination_to_index(
                    features_values_combination
                )"
                that index is then put into "train_indices" set.
        '''
        train_indices = set()
        frozen_combination_part_sizes \
            = self.features_sizes[:frozen_combination_part_length]
        varying_combination_part_sizes \
            = self.features_sizes[frozen_combination_part_length:]
        frozen_combination_part_generator = itertools.product(
            *[range(size) for size in frozen_combination_part_sizes]
        )

        num_varying_combination_part_variants \
            = int(np.prod(varying_combination_part_sizes))
        num_to_subsample_for_each_frozen_part = compute_proportion(
            self.train_val_split,
            num_varying_combination_part_variants
        )

        assert (
            num_to_subsample_for_each_frozen_part
                <= num_varying_combination_part_variants
        )

        for frozen_combination_part in frozen_combination_part_generator:

            indices_sampled_for_train = set(
                random.sample(
                    range(num_varying_combination_part_variants),
                    num_to_subsample_for_each_frozen_part
                )
            )

            varying_combination_part_generator = itertools.product(
                *[range(size) for size in varying_combination_part_sizes]
            )
            for i, varying_combination_part in enumerate(
                varying_combination_part_generator
            ):
                if i in indices_sampled_for_train:
                    train_indices.add(
                        self._features_classes_combination_to_index(
                            np.array(
                                frozen_combination_part
                                    + varying_combination_part
                            )
                        )
                    )

        return train_indices

    def _create_indices_for_feature_values(self):

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

        assert self.train_indices

        indices_for_feature_values = create_empty_indices_for_feature_values(
            self.possible_features,
            self.features_sizes,
            self.feature_to_idx
        )

        for features_classes_combination in itertools.product(
            *[range(size) for size in self.features_sizes]
        ):
            image_index = self._features_classes_combination_to_index(
                np.array(features_classes_combination)
            )
            split_name = (
                "train"
                    if image_index in self.train_indices
                    else "test"
            )
            for feature_idx, feature_value_idx in enumerate(
                features_classes_combination
            ):
                (indices_for_feature_values
                    [split_name]
                    [self.possible_features[feature_idx]]
                    [feature_value_idx]
                        .add(image_index)
                )

        return indices_for_feature_values

    def _features_classes_combination_to_index(
        self,
        features_classes_combination
    ):
        assert isinstance(features_classes_combination, np.ndarray)
        return np.dot(
            features_classes_combination,
            self.features_bases
        ).astype(int)

    def get_dataset(self, indices):
        """
        Args:
            indices (List[Set[int]]): a list which "c-th" element
                is a set of indices for images that belong to the class "c".
                Train/val split does not influence this method.
        """

        return DSpritesSingleLabelDataset(
            dsprites_images=self.imgs,
            indices_for_classes_to_extract=indices,
            colored=self.colored,
            transform=self.transform
        )

    def get_unlabelled_dataset(self, indices):
        """
        Args:
            indices (List[int]): a list of indices for elements
                that should form the dataset.
                Train/val split does not influence this method.
        """

        return DSpritesUnlabelledDataset(
            dsprites_images=self.imgs,
            indices_to_extract=indices,
            colored=self.colored,
            transform=self.transform
        )

    def get_multilabelled_offdiag_dataset(self, indices_to_labels):

        return DSpritesMultiLabelDataset(
            dsprites_images=self.imgs,
            indices_to_labels_to_extract=indices_to_labels,
            colored=self.colored,
            transform=self.transform
        )

    def _assert_dataset_specific_feature(self, feature):
        if feature == "color" and self.colored == False:
            raise Exception(
                "Can't use \"{}\" feature "
                "for non-colored dSprites dataset.".format(feature)
            )

    def _index_to_features_combination(self, index):
        assert self.features_bases
        return coefficients_for_bases(index, self.features_bases)


def make_dsprites_base_data(
    dsprites_config,
    logger=make_logger()
) -> BaseDataDSprites:

    log_or_print("Making dsprites_base_data..", logger=logger)
    dataset_name = "dsprites"
    dataset_path = os.path.join(dsprites_config["path"], dataset_name + ".npz")
    fetch_data(
        dataset_path=dataset_path,
        assert_folder_func=assert_dataset_as_file(
            get_dsprites_file_hash(dataset_name)
        ),
        do_fetch_data_func=do_fetch_dataset_as_file(
            get_dsprites_dataset_url(dataset_name)
        ),
        logger=logger,
        dataset_path_is_folder=False
    )
    dsprites_zip = read_dsprites_npz(
        filename=dataset_path
    )
    transform = make_transforms(dsprites_config.get("transforms"))
    return BaseDataDSprites(
        dsprites_zip=dsprites_zip,
        colored=(dsprites_config["type"] == "color"),
        train_val_split=dsprites_config["train_val_split"],
        transform=transform
    )


def load_dsprites_base_data(path):
    return pickle.load(open(path, "rb"))


def read_dsprites_npz(filename):
    return np.load(filename, allow_pickle=True, encoding='latin1')


# TODO(Alex | 08.09.2022) Think about better
# unlabeled data extraction? via method?
def get_dsprites_unlabeled_data(
    dataset_config,
    split,
    n_images,
    cache_path,
    logger
):
    base_data = make_or_load_from_cache(
        "base_data_dsprites",
        dataset_config,
        make_dsprites_base_data,
        load_dsprites_base_data,
        cache_path=cache_path,
        forward_cache_path=False,
        logger=logger
    )
    assert split in base_data.indices_for_feature_values.keys()
    return torch.utils.data.Subset(
        base_data.get_dataset(
            next(iter(base_data.indices_for_feature_values[split].values())),
            transform=T.Lambda(lambda x: torch.flatten(x))
        ),
        torch.linspace(
            0,
            n_images - 1,
            n_images,
            dtype=torch.int
        )
    )
