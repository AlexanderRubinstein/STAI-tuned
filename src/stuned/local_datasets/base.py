import numpy as np

class BaseData:

    '''
        input:
            <possible_features> - all features (aka factors)
                that can describe images
            <possible_features_values> - which values <possible_features> take
            <train_val_split> - proportion in which images
                are split into train/test
                following the rule that number of images
                taking each value from <possible_features_values>
                is approximately the same for each split

        defines hash function:

            self.indices_for_feature_values[<split>][<feature>][<value>]
                -> <indices>

                where <indices> point to images from <split> train/val split
                taking value <value> of feature <feature>
    '''

    def __init__(
        self,
        possible_features,
        possible_features_values,
        train_val_split
    ):
        self.possible_features = possible_features
        self.feature_to_idx = {}
        for feature_id, feature in enumerate(self.possible_features):
            self.feature_to_idx[feature] = feature_id
        self.possible_features_values = [
            possible_features_values[feature]
                for feature
                    in self.possible_features
        ]
        self.features_sizes = np.array(
            [len(values) for values in self.possible_features_values]
        )

        self._init_helpers()

        self.train_val_split = train_val_split
        self.train_indices = self._split_into_train_indices()
        self.total_num_samples = self._compute_total_num_samples()
        self.indices_for_feature_values \
            = self._create_indices_for_feature_values()

    def _init_helpers(self):
        pass

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
        raise NotImplementedError()

    def _create_indices_for_feature_values(self):
        raise NotImplementedError()

    def _compute_total_num_samples(self):
        raise NotImplementedError()
