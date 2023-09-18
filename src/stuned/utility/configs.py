import os
import warnings

from .constants import HARDCODED_CONFIG, AUTOGEN_PREFIX, EXP_NAME_CONFIG_KEY, START_TIME_CONFIG_KEY, \
    RUN_PATH_CONFIG_KEY, NESTED_CONFIG_KEY_SEPARATOR
# local modules
from .logger import make_logger
from .utils import (
    read_yaml,
    get_current_time,
    get_current_run_folder,
    get_hash,
    apply_func_to_dict_by_nested_key,
    normalize_path
)

if NESTED_CONFIG_KEY_SEPARATOR != '/':
    warnings.warn("=====================================================")
    warnings.warn("WARNING!!!! Arnas changed the `NESTED_CONFIG_KEY_SEPARATOR` from '/' to '{}'"
                  "\nPlease urge him to get his separators straight!".format(NESTED_CONFIG_KEY_SEPARATOR))
    warnings.warn("=====================================================")


def make_csv_config(
    csv_path,
    csv_row_number,
    spreadsheet_url,
    worksheet_name
):
    return {
        "path": csv_path,
        "row_number": csv_row_number,
        "spreadsheet_url": spreadsheet_url,
        "worksheet_name": worksheet_name
    }


def get_config(config_path, logger=None):

    if not os.path.exists(config_path):
        raise Exception(
            "Config was not found under this path: {}".format(
                config_path
            )
        )

    if logger is None:
        logger = make_logger()

    config_dirname = os.path.dirname(config_path)

    experiment_name = os.path.basename(config_dirname)
    if experiment_name == AUTOGEN_PREFIX:
        experiment_name = os.path.basename(os.path.dirname(config_dirname))
    logger.log(
        "Reading config for \"{}\" from {}".format(
            experiment_name,
            config_path
        ),
        auto_newline=True
    )

    experiment_config = read_yaml(config_path)
    if "use_hardcoded_config" in experiment_config and experiment_config["use_hardcoded_config"]:
        logger.log(
            "Using hardcoded config."
        )
        assert HARDCODED_CONFIG
        experiment_config = HARDCODED_CONFIG
        experiment_config.pop(EXP_NAME_CONFIG_KEY)
        experiment_config.pop(RUN_PATH_CONFIG_KEY)
        config_path = None

    experiment_config[EXP_NAME_CONFIG_KEY] = experiment_name

    experiment_config[START_TIME_CONFIG_KEY] = get_current_time()
    experiment_config[RUN_PATH_CONFIG_KEY] = get_current_run_folder(
        experiment_name,
        get_hash(experiment_config)
    )

    paths_in_config = find_nested_keys_by_keyword_in_config(
        experiment_config,
        "path"
    )
    normalize_paths(experiment_config, paths_in_config)

    return experiment_config


def find_nested_keys_by_keyword_in_config(
    config,
    keyword,
    separator=NESTED_CONFIG_KEY_SEPARATOR,
    prefix=""
):
    if isinstance(config, dict):
        res = []
        for key in config.keys():
            subpaths = find_nested_keys_by_keyword_in_config(
                config[key],
                keyword=keyword,
                separator=separator,
                prefix=prefix+key+separator
            )
            res.extend(subpaths)
        return res
    else:
        assert prefix
        assert prefix[-1] == separator
        if keyword in prefix:
            return [prefix[:-1]]
    return []


def normalize_paths(config, nested_keys, separator=NESTED_CONFIG_KEY_SEPARATOR):
    for key in nested_keys:
        apply_func_to_dict_by_nested_key(
            config,
            key.split(separator),
            normalize_path
        )
