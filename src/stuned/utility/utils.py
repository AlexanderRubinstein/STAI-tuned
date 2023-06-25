import torch
import numpy as np
import random
import os
import yaml
from datetime import datetime
import subprocess
from deepdiff import DeepDiff, model as dd_model
from hashlib import blake2b, md5
import pickle
import shutil
import socket
import traceback
import yaml
from tempfile import NamedTemporaryFile
import csv
from filelock import FileLock
import json
import time
import signal
import platform
import sys
from collections import Counter
import itertools
import contextlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import importlib


DEFAULT_HASH_SIZE = 10
DEFAULT_EXPERIMENTS_FOLDER = "experiments"
PROFILER_GROUP_BY_STACK_N = 5
PROFILER_OUTPUT_ROW_LIMIT = 10
DEFAULT_FILE_CHUNK_SIZE = 4096
EMPTY_CSV_TOKEN = "?"
DEFAULT_ENV_NAME = os.environ["DEFAULT_ENV"]
TEST_ENV_NAME = os.environ["TEST_ENV"]
BASHRC_PATH = os.path.join(os.environ["HOME"], ".bashrc")
NEW_SHELL_INIT_COMMAND = "source {} && conda activate".format(BASHRC_PATH)
FLOATING_POINT = "."
NAME_SEP = "_"
NULL_CONTEXT = contextlib.nullcontext()


DEFAULT_SLEEP_TIME = 10
DEFAULT_NUM_ATTEMTPS = 10
MAX_RETRIES_ERROR_MSG = "Maximum number of retries failed."


DEFAULT_INDENT_IN_JSON = 2


SYSTEM_PLATFORM = platform.system().lower()


LIST_START_SYMBOL = '['
LIST_END_SYMBOL = ']'
DELIMETER = ','
QUOTE_CHAR = '\"'
ESCAPE_CHAR = '\\'


def read_yaml(yaml_file):
    with open(yaml_file, 'r') as stream:
        return yaml.safe_load(stream)


def apply_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # to suppress warning
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_current_time():
    return datetime.now()


def raise_unknown(param, value, location=""):
    exception_msg = "Unknown {}".format(param)
    if location:
        exception_msg += " in {}".format(location)
    exception_msg += ": {}".format(value)
    raise Exception(exception_msg)


def append_dict(total_dict, current_dict, allow_new_keys=False):
    """
    Append leaves of possibly nested <current_dict>
    to leaf lists of possibly nested <total_dict>
    """

    def to_create_new_key(is_new_total_dict, allow_new_keys, key, total_dict):
        return is_new_total_dict or (allow_new_keys and key not in total_dict)

    is_new_total_dict = False
    if len(total_dict) == 0:
        is_new_total_dict = True
    for key, value in current_dict.items():
        if isinstance(value, dict):
            if to_create_new_key(
                is_new_total_dict,
                allow_new_keys,
                key,
                total_dict
            ):
                sub_dict = {}
                append_dict(sub_dict, value)
                total_dict[key] = sub_dict
            else:
                assert key in total_dict
                sub_dict = total_dict[key]
                assert isinstance(sub_dict, dict)
                append_dict(sub_dict, value)
                total_dict[key] = sub_dict
        else:
            if to_create_new_key(
                is_new_total_dict,
                allow_new_keys,
                key,
                total_dict
            ):
                total_dict[key] = [value]
            else:
                assert key in total_dict
                assert isinstance(total_dict[key], list)
                total_dict[key].append(value)


def check_element_in_iterable(
    iterable,
    element,
    element_name="element",
    iterable_name="iterable",
    reference=None
):
    try:
        assert element in iterable
    except Exception as e:
        exception_msg = "No {} \"{}\" in {}:\n{}".format(
                element_name,
                element,
                iterable_name,
                iterable
            )
        if reference is not None:
            exception_msg += "\nFor:\n {}".format(reference)
        raise Exception(exception_msg)


def check_dict(dict, required_keys, optional_keys=[], check_reverse=False):
    for key in required_keys:
        check_element_in_iterable(dict, key, "key", "dict")
    if check_reverse:
        allowed_keys = set(required_keys + optional_keys)
        for key in dict.keys():
            check_element_in_iterable(
                allowed_keys,
                key,
                "key",
                "set of allowed keys",
                reference=dict
            )


def runcmd(cmd, verbose=False, logger=None):

    try:
        cmd_output = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            shell=True
        )
        if verbose:
            msg = cmd_output.decode('utf-8', errors='ignore')
            if msg:
                if logger:
                    logger.log(msg)
                else:
                    print(msg)
    except subprocess.CalledProcessError as e:
        raise Exception(e.output.decode('utf-8', errors='ignore'))


def compute_dicts_diff(dict1, dict2, ignore_order=True):
    ddiff = DeepDiff(dict1, dict2, ignore_order=ignore_order)
    return ddiff


def randomly_subsample_indices_uniformly(total_samples, num_to_subsample):
    weights = torch.tensor(
        total_samples * [1.0 / total_samples], dtype=torch.float
    )
    return torch.multinomial(weights, num_to_subsample)


def deterministically_subsample_indices_uniformly(
    total_samples,
    num_to_subsample
):
    assert num_to_subsample <= total_samples, \
        "Try to subsample more samples than exist."
    return torch.linspace(
        0,
        total_samples - 1,
        num_to_subsample,
        dtype=torch.int
    )


def get_device(use_gpu, idx=0):
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda:{}".format(idx))
        else:
            raise Exception("Cuda is not available.")
    else:
        return torch.device("cpu")


def get_model_device(model):
    assert isinstance(model, torch.nn.Module)
    return next(model.parameters()).device


def get_project_root_path():
    return os.path.abspath(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
    )


def make_unique_run_name(hashed_config_diff):
    return "{}---{}---{}".format(
        get_current_time(),
        hashed_config_diff,
        os.getpid()
    ).replace(" ", "_")


def get_current_run_folder(experiment_name, hashed_config_diff):
    return os.path.join(
        get_project_root_path(),
        DEFAULT_EXPERIMENTS_FOLDER,
        experiment_name,
        make_unique_run_name(hashed_config_diff)
    )


def get_hash(input_object, hash_size=DEFAULT_HASH_SIZE):
    h = get_hasher("blake2b", hash_size=hash_size)
    h.update(input_object.__repr__().encode())
    return h.hexdigest()


def get_value_from_config(file_path, value_name):
    with open(file_path, "r") as f:
        for line in f.readlines():
            line_elements = line.split()
            if len(line_elements) == 0:
                continue
            if value_name == line_elements[0]:
                if len(line_elements) < 2:
                    raise Exception(
                        "Line \"{}\" does not have value"
                        " for \"{}\".".format(line, value_name)
                    )
                return line_elements[1]
    raise Exception(
        "File \"{}\" does not contain"
        " \"{}\".".format(file_path, value_name)
    )





def get_checkpoint(experiment_config, logger=None):
    checkpoint_config = experiment_config["checkpoint"]
    if (
        "starting_checkpoint_path" in checkpoint_config
            and checkpoint_config["starting_checkpoint_path"] is not None
    ):
        checkpoint = init_checkpoint(
            starting_checkpoint_path
                =checkpoint_config["starting_checkpoint_path"],
            check_only_model=checkpoint_config["check_only_model"],
            logger=logger
        )
    else:
        checkpoint = init_checkpoint(
            experiment_name=experiment_config["experiment_name"],
            logger=logger
        )
    return checkpoint


def init_empty_checkpoint(experiment_name):
    return {
        "experiment_name": experiment_name,
        "current_epoch": 0,
        "model": None,
        "optimizer": None,
        "lr_scheduler": None,
        "criterion": None
    }


def init_checkpoint(
    starting_checkpoint_path=None,
    check_only_model=False,
    experiment_name=None,
    logger=None
):
    # checkpoint contents:
    #   - experiment_name
    #   - current epoch number
    #   - model
    #   - optimizer
    #   - lr_scheduler
    #   - criterion
    if starting_checkpoint_path is not None:
        log_or_print(
            "Reading checkpoing from {}..".format(starting_checkpoint_path),
            logger=logger,
            auto_newline=True
        )
        checkpoint = read_checkpoint(starting_checkpoint_path)
    else:
        assert experiment_name is not None
        checkpoint = init_empty_checkpoint(
            experiment_name
        )
    check_checkpoint(checkpoint, check_only_model)
    return checkpoint


def log_or_print(msg, logger=None, auto_newline=False):
    if logger:
        logger.log(msg, auto_newline=auto_newline)
    else:
        print(msg)


def error_or_print(msg, logger=None, auto_newline=False):
    if logger:
        logger.error(msg, auto_newline=auto_newline)
    else:
        print(msg, file=sys.stderr)


def read_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = pickle.load(open(checkpoint_path, "rb"))
    else:
        raise Exception(
            "Checkpoint path does not exist: {}".format(checkpoint_path)
        )
    return checkpoint


def check_checkpoint(checkpoint, check_only_model):
    if check_only_model:
        check_dict(checkpoint,
            [
                "model"
            ]
        )
    else:
        check_dict(checkpoint,
            [
                "experiment_name",
                "current_epoch",
                "model",
                "optimizer",
                "lr_scheduler",
                "criterion"
            ],
            check_reverse=True
        )


def save_checkpoint(
    checkpoint,
    checkpoint_folder,
    checkpoint_name=None,
    logger=None
):
    if checkpoint_name is None:
        checkpoint_name = make_checkpoint_name(checkpoint)
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_savepath = os.path.join(checkpoint_folder, checkpoint_name)
    log_msg = "Saving checkpoint to {}".format(checkpoint_savepath)
    if logger:
        logger.log(log_msg, auto_newline=True)
    else:
        print(log_msg)
    for obj in checkpoint.values():
        prepare_for_pickling(obj)
    pickle.dump(
        checkpoint,
        open(checkpoint_savepath, "wb")
    )


def prepare_for_pickling(obj):
    if hasattr(obj, "_prepare_for_pickling"):
        obj._prepare_for_pickling()


def make_checkpoint_name(checkpoint):
    return "checkpoint-epoch_{}.pkl".format(
        checkpoint["current_epoch"] + 1,
    )


def get_leaves_of_nested_dict(
    nested_dict,
    nested_key_prefix=[],
    include_values=False
):
    assert len(nested_dict) > 0, "Nested dict is empty."
    result = []
    for key, value in nested_dict.items():
        current_prefix = nested_key_prefix + [key]
        if isinstance(value, dict):
            result.extend(
                get_leaves_of_nested_dict(
                    value,
                    nested_key_prefix=current_prefix,
                    include_values=include_values
                )
            )
        else:
            if include_values:
                result.append((current_prefix, value))
            else:
                result.append(current_prefix)

    return result


def update_dict_by_nested_key(
    input_dict,
    nested_key_as_list,
    new_value,
    to_create_new_elements=False
):
    apply_func_to_dict_by_nested_key(
        input_dict,
        nested_key_as_list,
        lambda x: new_value,
        to_create_new_elements
    )


def apply_func_to_dict_by_nested_key(
    input_dict,
    nested_key_as_list,
    func,
    to_create_new_elements=False
):
    if not isinstance(input_dict, dict):
        raise Exception(
            "\"{}\" is expected to be a dict "
            "containing the following nested dicts: \n{}".format(
                input_dict,
                nested_key_as_list[:-1]
            )
        )
    if not isinstance(nested_key_as_list, list):
        raise Exception(
            "\"{}\" is expected "
            "to be a list of nested keys for dict: \n{}".format(
                nested_key_as_list,
                input_dict
            )
        )
    if len(nested_key_as_list) == 0:
        raise Exception("Empty nested key was given for dict update.")
    else:

        current_key = nested_key_as_list[0]
        is_leaf = (len(nested_key_as_list) == 1)

        if not current_key in input_dict:
            if to_create_new_elements:
                if is_leaf:
                    input_dict[current_key] = None
                else:
                    input_dict[current_key] = {}
            else:
                raise Exception(
                    "During recursive dict update {} "
                    "was not found in {}".format(
                        current_key,
                        input_dict
                    )
                )

        if is_leaf:
            input_dict[current_key] \
                = func(input_dict[current_key])
        else:
            new_subdict = input_dict[current_key]
            apply_func_to_dict_by_nested_key(
                new_subdict,
                nested_key_as_list[1:],
                func,
                to_create_new_elements
            )
            input_dict[current_key] = new_subdict


def extract_profiler_results(prof):
    return prof.key_averages(group_by_stack_n=PROFILER_GROUP_BY_STACK_N).table(
        sort_by="self_cuda_memory_usage",
        row_limit=PROFILER_OUTPUT_ROW_LIMIT
    )


def check_equal_shape(tensors_list):
    shape = None
    for tensor in tensors_list:
        if shape:
            assert tensor.shape == shape
        else:
            shape = tensor.shape


def move_folder_contents(src_folder, dst_folder):
    for file_or_folder in os.listdir(src_folder):
        dst_path = os.path.join(dst_folder, file_or_folder)
        assert not os.path.exists(dst_path)
        os.rename(
            os.path.join(src_folder, file_or_folder),
            dst_path
        )


def remove_all_but_subdirs(src_folder, objects_to_keep):
    for object_name in objects_to_keep:
        assert os.path.exists(os.path.join(src_folder, object_name))
    objects_to_keep = set(objects_to_keep)
    objects_in_folder_before = os.listdir(src_folder)
    for object_name in objects_in_folder_before:
        if not object_name in objects_to_keep:
            remove_file_or_folder(os.path.join(src_folder, object_name))
    assert len(objects_to_keep) == len(os.listdir(src_folder))


def remove_file_or_folder(file_or_folder):
    if os.path.isfile(file_or_folder):
        os.remove(file_or_folder)
    elif os.path.isdir(file_or_folder):
        shutil.rmtree(file_or_folder)
    else:
        raise_unknown(
            "file type",
            "",
            "for object \"{}\"".format(file_or_folder)
        )


def is_nested_dict(input_dict):
    for value in input_dict.values():
        if isinstance(value, dict):
            return True
    return False


def ensure_separator_after_folder(folder_path):
    assert os.path.isdir(folder_path)
    if folder_path[-1] != os.path.sep:
        folder_path += os.path.sep
    return folder_path


def compute_file_hash(
    filename,
    chunksize=DEFAULT_FILE_CHUNK_SIZE,
    hash_type="md5"
):
    hasher = get_hasher(hash_type)
    with open(filename, "rb") as f:
        for byte_block in iter(
            lambda: f.read(chunksize),
            b""
        ):
            hasher.update(byte_block)
    return hasher.hexdigest()


def get_hasher(hash_type, hash_size=DEFAULT_HASH_SIZE):
    if hash_type == "md5":
        return md5()
    elif hash_type == "blake2b":
        return blake2b(digest_size=hash_size)
    else:
        raise_unknown("hash type", hash_type, "getting hasher")


def get_hostname():
    return socket.gethostname()


def remove_filename_extension(filename, must_have_extension=True):
    if "." in filename:
        filename = filename[:filename.find(".")]
        assert "." not in filename
    elif must_have_extension:
        raise Exception(
            "Filename {} is expected to have extension".format(filename)
        )
    return filename


def range_for_each_group(num_groups, num_elements):

    assert num_elements >= num_groups

    indices_per_group = int(num_elements / num_groups)
    remainder = num_elements % num_groups

    return [
        (
            group_id * indices_per_group
                + (
                    min(group_id, remainder)
                ),
            (group_id + 1) * (indices_per_group)
                + (
                    min(group_id, remainder - 1) + 1
                )
        ) for group_id in range(num_groups)
    ]


def coefficients_for_bases(number, bases):
    combination = []
    prev_base = float("Inf")
    for base in bases:
        assert prev_base > base
        current_value = int(number / base)
        combination.append(current_value)
        number = number % base
        if len(combination) < len(bases):
            prev_base = base
        else:
            assert number == 0
    return combination


def error_callback(exception):
    raise Exception("".join(traceback.format_exception(exception)))


def deterministic_subset(input_set, num_to_subsample):
    assert num_to_subsample <= len(input_set)
    subsampled_idxs = np.linspace(
        0,
        len(input_set) - 1,
        num_to_subsample,
        dtype=np.int32
    )
    res = set()
    set_as_list = sorted(list(input_set))
    for i in subsampled_idxs:
        res.add(set_as_list[i])
    return res


def compute_proportion(proportion, total_number):
    return max(1, int(
        proportion * total_number
    ))


def make_autogenerated_config_name(input_csv_path, row_number):
    return "[{}]_{}_autogenerated_config.yaml".format(
        row_number,
        get_hash(input_csv_path)
    )


def save_as_yaml(output_file_name, data):
    with open(output_file_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def write_into_csv_with_column_names(
    file_path,
    row_number,
    column_name,
    value,
    delimiter=DELIMETER,
    quotechar=QUOTE_CHAR,
    quoting=csv.QUOTE_MINIMAL,
    escapechar=ESCAPE_CHAR,
    doublequote=False,
    replace_nulls=False,
    append_row=False,
    use_lock=True
):
    """
    Insert <value> into the intersection of row number <row_number>
    and column <column_name> of the csv file with path <file_path>.


    Args:

        file_path (str): path of csv file in filesystem.

        row_number (int): number of row in which <value> should be inserted.
            Rows numbering starts with 0. Row 0 is the row with column names.
            Therefore if <row_number> equals 0, column renaming will take place
            (but it is possible only for a non-empty file).
            For an empty file <row_number> should be 1,
            as a new column will be created anyway.

        column_name (str): column in which <value> should be inserted.
            If the csv file does not contain <column_name>,
            new column with the name <column_name> is created.
            When new column is created row <row_number> is filled with <value>
            other rows are filled with "EMPTY_CSV_TOKEN" if they exist.

        value (Any): value to insert into the csv file.
            It is converted to string before insertion.

        other args
    """

    tempfile = NamedTemporaryFile('w+t', newline='', delete=False)

    lock = make_file_lock(file_path) if use_lock else NULL_CONTEXT

    value = as_str_for_csv(value)

    with lock:

        with open(file_path, 'r', newline='') as csv_file, tempfile:

            reader = csv.reader(
                (
                    (x.replace('\0', '') for x in csv_file)
                        if replace_nulls else csv_file
                ),
                delimiter=delimiter,
                quotechar=quotechar,
                quoting=quoting,
                escapechar=escapechar,
                doublequote=doublequote
            )
            writer = csv.writer(
                tempfile,
                delimiter=delimiter,
                quotechar=quotechar,
                quoting=quoting,
                escapechar=escapechar,
                doublequote=doublequote
            )

            appended_column = False
            value_inserted = False

            num_rows = count_rows_in_file(csv_file)
            num_cols = None

            pos_in_row = None

            if num_rows == 0:

                assert row_number == 1, \
                    (
                        "Can't insert into row number {} of empty file, "
                        "only row number 1 is possible."
                    ).format(row_number)

                writer.writerow([column_name])
                append_row = True
                num_cols = 1
                num_rows += 1

            else:
                for current_row_number, row in enumerate(reader):

                    if current_row_number == 0:
                        if column_name in row:
                            pos_in_row = row.index(column_name)
                        else:
                            row.append(column_name)
                            appended_column = True
                            pos_in_row = len(row) - 1

                    if current_row_number == row_number:
                        assert pos_in_row is not None
                        if appended_column and row_number > 0:
                            row.append(value)
                        else:
                            assert len(row) > pos_in_row, \
                                "CSV's contents are inconsistent " \
                                "with the number of columns " \
                                "for the file {}".format(
                                    file_path
                                )
                            row[pos_in_row] = value
                        value_inserted = True
                    elif appended_column and current_row_number > 0:
                        row.append(EMPTY_CSV_TOKEN)

                    writer.writerow(row)
                    num_cols = len(row)

            if append_row:

                assert num_cols is not None and num_cols > 0

                assert row_number == num_rows

                writer.writerow([value] + [EMPTY_CSV_TOKEN] * (num_cols - 1))
                value_inserted = True

        if not value_inserted:
            raise Exception(
                "CSV file {} has {} rows, while insertion "
                "into row {} was requested!".format(
                    file_path,
                    num_rows,
                    row_number
                )
            )

        shutil.move(tempfile.name, file_path)


def count_rows_in_file(file):

    rowcount = 0

    for _ in file:
        rowcount += 1

    file.seek(0)

    return rowcount


def remove_elements_from_the_end(sequence, element_to_remove):

    assert len(sequence)

    end_pos = len(sequence) - 1

    while end_pos >= 0 and sequence[end_pos] == element_to_remove:
        end_pos -= 1

    return sequence[:end_pos + 1]


def itself_and_lower_upper_case(word):
    return [word, word.lower(), word.upper()]


def decode_strings_in_dict(
    input_dict,
    list_separators,
    list_start_symbol,
    list_end_symbol
):

    for key, value in input_dict.items():

        if isinstance(value, str):

            if value == "":
                continue

            value = decode_val_from_str(
                value,
                list_separators,
                list_start_symbol,
                list_end_symbol
            )

            input_dict[key] = value


def decode_val_from_str(
    value,
    list_separators=[' '],
    list_start_symbol='[',
    list_end_symbol=']'
):

    if isinstance(value, str):
        value = value.strip()

    if str_is_number(value):

        value = parse_float_or_int_from_string(value)

    elif (
        len(value) > 1
            and (value[0] == list_start_symbol
            or value[-1] == list_end_symbol)
    ):

        assert (
            value[0] == list_start_symbol
                and value[-1] == list_end_symbol
        ), f"Possibly incomplete list expression: {value}"

        value = parse_list_from_string(
            value,
            list_separators=list_separators,
            list_start_symbol=list_start_symbol,
            list_end_symbol=list_end_symbol
        )

    elif (
        value in itself_and_lower_upper_case("None")
            or value in itself_and_lower_upper_case("Null")
    ):
        value = None

    elif value in itself_and_lower_upper_case("False"):
        value = False

    elif value in itself_and_lower_upper_case("True"):
        value = True

    return value


def replace_many_by_one(
    input_string,
    items_to_replace,
    value_to_insert
):
    # TODO(Alex | 25.02.2023) do in O(n) with regex
    for item_to_replace in items_to_replace:

        if item_to_replace == value_to_insert:
            continue

        input_string = input_string.replace(
            item_to_replace,
            value_to_insert
        )

    return input_string


def str_is_number(input_str):

    exponential = False
    has_floating_point = False
    for i in range(len(input_str)):
        if input_str[i].isdigit():
            continue
        if input_str[i] == '-':
            if i == 0:
                continue
            elif exponential and input_str[i - 1] == 'e':
                continue
        if input_str[i] == '+' and exponential and input_str[i - 1] == 'e':
            continue
        if input_str[i] == 'e' and not exponential:
            exponential = True
            continue
        if input_str[i] == FLOATING_POINT and not has_floating_point:
            has_floating_point = True
            continue
        return False

    if exponential and not input_str[-1].isdigit():
        return False

    return True


def parse_float_or_int_from_string(value_as_str):
    if FLOATING_POINT in value_as_str:
        return float(value_as_str)
    else:
        return int(value_as_str)


def parse_list_from_string(
    value_as_str,
    list_separators,
    list_start_symbol=LIST_START_SYMBOL,
    list_end_symbol=LIST_END_SYMBOL
):

    assert len(list_separators)
    assert len(value_as_str) > 1
    assert (
        value_as_str[0] == list_start_symbol
            and value_as_str[-1] == list_end_symbol
    )

    if len(value_as_str) == 2:
        return []
    else:

        value_as_str = value_as_str[1:-1]

        replace_many_by_one(
            value_as_str,
            list_separators,
            list_separators[0]
        )

        result_list = value_as_str.split(list_separators[0])

        for i in range(len(result_list)):

            result_list[i] = decode_val_from_str(
                result_list[i],
                list_separators,
                list_start_symbol,
                list_end_symbol
            )

        return result_list


def read_csv_as_dict(
    csv_path,
    delimeter=DELIMETER,
    quotechar=QUOTE_CHAR,
    quoting=csv.QUOTE_MINIMAL,
    escapechar=ESCAPE_CHAR,
    doublequote=False
):

    result = {}

    with open(csv_path, newline='') as input_csv:
        csv_reader = csv.DictReader(
            input_csv,
            delimiter=delimeter,
            quotechar=quotechar,
            quoting=quoting,
            escapechar=escapechar,
            doublequote=doublequote
        )

        result[0] = {}

        for fieldname in csv_reader.fieldnames:
            result[0][fieldname] = fieldname

        for csv_row_number, csv_row in enumerate(csv_reader):
            result[csv_row_number + 1] = csv_row

    return result


def normalize_path(path, current_dir=get_project_root_path()):
    if path is None:
        return None
    assert isinstance(path, str)
    assert path
    path = os.path.expanduser(path)
    if path[0] == '.':
        path = os.path.join(current_dir, path)
    return os.path.abspath(path)


def read_json(json_path):
    with open(json_path, 'r') as f:
        json_contents = json.load(f)
        return json_contents


def dicts_with_non_intersecting_keys(dict_one, dict_two):
    return len(set(dict_one.keys()).intersection(set(dict_two.keys()))) == 0


def is_number(x):
    return str_is_number(str(x))


def assert_two_values_are_close(value_1, value_2, **isclose_kwargs):

    def assert_info(value_1, value_2):
        return "value 1: {}\n\nvalue 2: {}".format(value_1, value_2)

    def assert_is_close(value_1, value_2, isclose_func, **isclose_kwargs):
        assert isclose_func(value_1, value_2, **isclose_kwargs).all(), \
            assert_info(value_1, value_2)

    if value_1 is None:
        assert value_2 is None

    if not (is_number(value_1) and is_number(value_2)):
        value_1_type = type(value_1)
        value_2_type = type(value_2)
        assert value_1_type == value_2_type, \
            assert_info(value_1_type, value_2_type)

    if isinstance(value_1, (list, tuple, dict)):
        assert len(value_1) == len(value_2), assert_info(value_1, value_2)
        if isinstance(value_1, dict):
            iterable = zip(
                sorted(value_1.items(), key=(lambda x: x[0])),
                sorted(value_2.items(), key=(lambda x: x[0]))
            )
        else:
            iterable = zip(value_1, value_2)
        for subvalue_1, subvalue_2 in iterable:
            assert_two_values_are_close(
                subvalue_1,
                subvalue_2,
                **isclose_kwargs
            )
    elif isinstance(value_1, np.ndarray):
        assert_is_close(value_1, value_2, np.isclose, **isclose_kwargs)
    elif torch.is_tensor(value_1):
        assert_is_close(value_1, value_2, torch.isclose, **isclose_kwargs)
    else:
        assert value_1 == value_2, assert_info(value_1, value_2)


def bootstrap_by_key_subname(input_dict, subname_to_bootstrap):
    """
    Recursively lift dict entries which contain <subname_to_bootstrap>
    in their key on the upper level from leaves to root.

    Example:
        <subname_to_bootstrap> = "subkey"
        input_dict = {
            "a": 1,
            "b": {
                "csubkeyd": {
                    "e": 2,
                    "gsubkeyh": 3
                },
                "f": 4
            }
        }
        result = {
            "a": 1,
            "b": 3
        }
    """
    new_key_values = []
    for key, value in input_dict.items():
        if isinstance(value, dict):
            bootstrap_by_key_subname(value, subname_to_bootstrap)
            key_of_interest = find_by_subkey(value.keys(), subname_to_bootstrap)
            if key_of_interest is not None:
                new_key_values.append((key, value[key_of_interest]))
    for key, new_value in new_key_values:
        input_dict[key] = new_value


def find_by_subkey(iterable, subkey, assert_found=False):
    for key in iterable:
        if isinstance(key, str) and subkey in key:
            return key
    if assert_found:
        assert False, \
            "Key with subkey {} wasn't found in {}.".format(subkey, iterable)
    return None


def get_system_root_path():
    return os.path.abspath(os.sep)


def prepare_factory_without_args(func, **kwargs):

    def factory_without_args():
        return func(**kwargs)

    return factory_without_args


def raise_func(logger):
    error_or_print(MAX_RETRIES_ERROR_MSG, logger)
    raise


def retrier_factory(
    logger=None,
    final_func=raise_func,
    max_retries=DEFAULT_NUM_ATTEMTPS,
    sleep_time=DEFAULT_SLEEP_TIME,
    infer_logger_from_args=None
):

    assert (
            logger != "auto" and infer_logger_from_args is None
        or
            logger == "auto" and infer_logger_from_args is not None
    ), "Provide \"infer_logger_from_args\" iff \"logger == auto\""

    def retrier(func):

        # without this line logger is not seen in wrapped_func
        logger_in_retrier = logger

        def wrapped_func(*args, **kwargs):

            logger = logger_in_retrier

            if logger == "auto":
                logger = infer_logger_from_args(*args, **kwargs)

            num_attempts = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except:
                    if num_attempts >= max_retries:
                        return final_func(logger)
                    num_attempts += 1
                    time.sleep(sleep_time)
                    retry_print = None
                    if logger is not None and hasattr(logger, "retry_print"):
                        retry_print = logger.retry_print
                        logger.retry_print = False
                    retrying_msg = "{}\nRetrying {}: {}/{}.\n\n".format(
                        traceback.format_exc(),
                        func.__name__,
                        num_attempts,
                        max_retries
                    )
                    try:
                        error_or_print(
                            retrying_msg,
                            logger
                        )
                    except:
                        print(retrying_msg)
                    if retry_print is not None:
                        logger.retry_print = retry_print

        return wrapped_func

    return retrier


# from: https://stackoverflow.com/questions/8230315/how-to-json-serialize-sets
class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (set, dd_model.PrettyOrderedSet)):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


# from: https://www.tensorflow.org/tensorboard/text_summaries
def pretty_json(hp, cls=SetEncoder, default=str):
    json_hp = json.dumps(
        hp,
        indent=DEFAULT_INDENT_IN_JSON,
        cls=cls,
        default=default
    )
    return "".join("\t" + line for line in json_hp.splitlines(True))


def touch_file(file_path):
    open(file_path, 'a').close()
    return file_path


def cat_or_assign(accumulating_tensor, new_tensor):
    if accumulating_tensor is None:
        return new_tensor
    return torch.cat((accumulating_tensor, new_tensor))


def kill_processes(processes_to_kill, logger=None):
    for process_to_kill in processes_to_kill:
        try:
            os.kill(
                process_to_kill,
                signal.SIGTERM
            )
        except Exception:
            error_or_print(traceback.format_exc(), logger)


def read_old_checkpoint(checkpoint_path):
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "train_eval")
    )
    checkpoint = read_checkpoint(checkpoint_path)
    sys.path.pop(0)
    return checkpoint


def read_model_from_old_checkpoint(path):
    checkpoint = read_old_checkpoint(path)
    return checkpoint["model"]


def make_file_lock(file_name):
    return FileLock("{}.lock".format(file_name))


def check_duplicates(input_list):

    counter_dict = dict(Counter(input_list))
    if max(counter_dict.values()) > 1:
        raise Exception(f"Duplicates in:\n{input_list}\nCounts:\n{counter_dict}")


def has_nested_attr(object, nested_attr):
    assert len(nested_attr) > 0
    if len(nested_attr) == 1:
        return hasattr(object, nested_attr[0])
    else:
        return (
            hasattr(object, nested_attr[0])
                and has_nested_attr(
                    getattr(object, nested_attr[0]),
                    nested_attr[1:]
                )
        )


def get_nested_attr(object, nested_attr):
    assert len(nested_attr) > 0
    if len(nested_attr) == 1:
        return getattr(object, nested_attr[0])
    else:
        return (
            get_nested_attr(getattr(object, nested_attr[0]), nested_attr[1:])
        )


def set_nested_attr(object, nested_attr, value):
    assert len(nested_attr) > 0
    if len(nested_attr) == 1:
        return setattr(object, nested_attr[0], value)
    else:
        set_nested_attr(getattr(object, nested_attr[0]), nested_attr[1:], value)


def write_csv_dict_to_csv(dict_from_csv, csv_file, **kwargs):

    if os.path.exists(csv_file):
        remove_file_or_folder(csv_file)

    touch_file(csv_file)

    for row_number, row_as_dict in dict_from_csv.items():
        if row_number == 0:
            continue
        for i, (column_name, value) in enumerate(row_as_dict.items()):
            write_into_csv_with_column_names(
                csv_file,
                row_number,
                column_name,
                value,
                append_row=(i == 0),
                **kwargs
            )


def expand_csv(
    csv_to_expand,
    expanded_csv,
    expansion_start_symbol='{',
    expansion_end_symbol='}',
    expansion_delimeter=" | ",
    range_start_symbol='<',
    range_end_symbol='>',
    range_delimeter=' '
):

    def expand_row(
        row_as_dict,
        expansion_start_symbol,
        expansion_end_symbol,
        expansion_delimeter,
        range_start_symbol,
        range_end_symbol,
        range_delimeter
    ):

        def build_range(range_as_str, range_delimeter):
            range_as_str_split = range_as_str.split(range_delimeter)
            assert len(range_as_str_split) == 4
            for i in range(len(range_as_str_split)):
                range_as_str_split[i] \
                    = decode_val_from_str(range_as_str_split[i])

            start = range_as_str_split[0]
            end = range_as_str_split[1]
            step = range_as_str_split[2]
            log_scale = range_as_str_split[3]
            result = []

            assert step > 1 if log_scale else step > 0, \
                "Range should be increasing"

            while start < end:

                result.append(start)

                if log_scale:
                    start *= step
                else:
                    start += step

            return result

        keys = list(row_as_dict.keys())
        for key in keys:
            value = row_as_dict[key]
            if isinstance(value, str):
                value = value.strip()
            if (
                isinstance(value, str)
                    and len(value) >= 2
                    and value[0] == expansion_start_symbol
                    and value[-1] == expansion_end_symbol
            ):
                if (
                    len(value) >= 4
                    and value[1] == range_start_symbol
                    and value[-2] == range_end_symbol
                ):
                    assert len(value) > 4, "Empty range expression"
                    assert range_delimeter in value, \
                        f"Range expression without range delimeter {value}"
                    value = (
                        expansion_start_symbol
                            + expansion_delimeter.join(
                                str(el) for el in build_range(
                                    value[2:-2],
                                    range_delimeter
                                )
                            )
                            + expansion_end_symbol
                    )

                row_as_dict[key] = parse_list_from_string(
                    value,
                    [expansion_delimeter],
                    list_start_symbol=expansion_start_symbol,
                    list_end_symbol=expansion_end_symbol
                )
            else:
                row_as_dict[key] = [value]

        cross_product = itertools.product(*[row_as_dict[key] for key in keys])

        result = []
        for values in cross_product:
            result.append({key: value for key, value in zip(keys, values)})

        return result

    dict_to_expand = read_csv_as_dict(csv_to_expand, quotechar=QUOTE_CHAR)
    expanded_dict = {}
    final_row_id = 0

    for row_as_dict in dict_to_expand.values():

        expanded_rows = expand_row(
            row_as_dict,
            expansion_start_symbol=expansion_start_symbol,
            expansion_end_symbol=expansion_end_symbol,
            expansion_delimeter=expansion_delimeter,
            range_start_symbol=range_start_symbol,
            range_end_symbol=range_end_symbol,
            range_delimeter=range_delimeter
        )
        for row in expanded_rows:
            expanded_dict[final_row_id] = row
            final_row_id += 1

    write_csv_dict_to_csv(expanded_dict, expanded_csv)


def instantiate_from_config(config, object_key_in_config, make_func, logger):
    object_config = config.get(object_key_in_config)
    if object_config is None:
        return None
    else:
        return make_func(object_config, logger)


class TimeStampEventHandler(FileSystemEventHandler):

    def __init__(self):
        super(TimeStampEventHandler, self).__init__()
        self.update_time()

    def on_any_event(self, event):
        self.update_time()

    def update_time(self):
        self.last_change_time = get_current_time()

    def has_events(self, delta):
        time_since_last_change = (
            get_current_time() - self.last_change_time
        ).total_seconds()
        if time_since_last_change <= delta:
            return True
        else:
            return False


def folder_still_has_updates(path, delta, max_time, check_time=None):
    """
    Check every <check_time> seconds whether <path> had any updates (events).
    Observe the <path> for at most <max_time>.
    If there were no updates for <delta> seconds return True, otherwise return
    False. If watchdog observer failed to start return None.
    """

    if check_time is None:
        check_time = delta

    n = max(1, int(max_time / check_time))
    i = 0
    event_handler = TimeStampEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)

    try:
        observer.start()
    except:
        return None

    has_events_bool = event_handler.has_events(delta)
    while has_events_bool and i < n:
        time.sleep(check_time)
        i += 1
        has_events_bool = event_handler.has_events(delta)

    observer.stop()
    observer.join()

    return has_events_bool


# https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/util.py#L88
def import_from_string(string, reload=False):
    if '.' in string:
        module, cls = string.rsplit('.', 1)
    else:
        module = string
        cls = None
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    module_imp = importlib.import_module(module, package=None)
    if cls is None:
        return module_imp
    else:
        return getattr(module_imp, cls)


def as_str_for_csv(value):
    if value is None:
        value = "None"
    value = str(value)
    return value


def check_consistency(
    first,
    second,
    first_consistent_group,
    second_consistent_group
):
    def make_msg(
        first,
        first_consistent_group,
        second,
        second_consistent_group
    ):
        return (
            "{} is from {}, therefore {} "
            "should be from {}."
        ).format(
            first,
            first_consistent_group,
            second,
            second_consistent_group
        )

    try:
        if first in first_consistent_group:
            exception_msg = make_msg(
                first,
                first_consistent_group,
                second,
                second_consistent_group
            )
            assert second in second_consistent_group
        if second in second_consistent_group:
            exception_msg = make_msg(
                second,
                second_consistent_group,
                first,
                first_consistent_group
            )
            assert first in first_consistent_group
    except AssertionError as e:
        raise Exception(exception_msg)
