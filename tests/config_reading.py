import os
import unittest
from unittest import mock

import stuned.utility.utils as utils
import stuned.utility.configs as configs

TEST_CONFIG_PATH = "test_config.yaml"


class TestPathNormalization(unittest.TestCase):

    def setUp(self):
        self.config = utils.read_yaml(TEST_CONFIG_PATH)

    def test_path_normalization_same_for_str_and_list(self):
        str_keys = ["path0", "path1", "path2", "path3"]
        list_key = "paths"

        # Check that paths are found in config
        paths_in_config = configs.find_nested_keys_by_keyword_in_config(
            self.config,
            "path",
        )
        self.assertEqual(
            paths_in_config,
            str_keys + [list_key],
        )

        configs.normalize_paths(self.config, paths_in_config)

        str_paths = [self.config[key] for key in str_keys]
        list_paths  = self.config[list_key]

        # Check that normalization does to lists the same as to strings
        for p_str, p_list in zip(str_paths, list_paths):
            self.assertEqual(p_str, p_list)
