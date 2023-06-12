import sys
import traceback
import multiprocessing as mp


# local modules
from .logger import (
    BaseLogger,
    make_logger
)
from .utils import (
    compute_dicts_diff,
    read_csv_as_dict
)


DEFAULT_N_PROCESSES = min(24, mp.cpu_count())


def wrapper_for_test(func, with_extended_args=False):

    def wrapped_func(*args, **kwargs):
        logger = make_logger()
        try:
            test_name = str(func).split()[1]
            logger.info("Starting test \"{}\"".format(test_name))
            with mp.Pool(DEFAULT_N_PROCESSES) as pool:
                with mp.Manager() as shared_memory_manager:
                    extended_args = {
                        "logger": logger,
                        "pool": pool,
                        "shared_memory_manager": shared_memory_manager
                    }
                    if with_extended_args:
                        func(*args, **kwargs, extended_args=extended_args)
                    else:
                        func(*args, **kwargs)
                    logger.info(f"{test_name}: OK")
        except:
            logger.error(traceback.format_exc())
            logger.error(f"{test_name}: Fail")
            sys.exit(1)

    return wrapped_func


def wrapper_for_test_with_extended_args(func):
    return wrapper_for_test(func, with_extended_args=True)


def expect_failed(
    func,
    fail_info=None,
    unrecoverable_error=RuntimeError,
    expected_error=Exception,
    logger=make_logger()
):
    fail_message = "Test is expected to fail"

    def wrapped_func(*args, **kwargs):
        try:
            if fail_info:
                fail_message = fail_info["msg"]

            func(*args, **kwargs)

            raise unrecoverable_error(fail_message)
        except expected_error as e:
            if isinstance(e, unrecoverable_error):
                raise e
            else:
                logger.info(
                    "Caught expected error ({}): \n\"{}\"".format(
                        fail_message,
                        e
                    ),
                    auto_newline=True
                )

    return wrapped_func


class DummyLogger(BaseLogger):

    def __init__(self):
        self.reset()
        self.retry_print = True

    def log(self, msg, auto_newline=False):
        self.log_strings += msg

    def error(self, msg, auto_newline=False):
        self.error_strings += msg

    def reset(self):
        self.log_strings = ""
        self.error_strings = ""


def make_dummy_logger():
    return DummyLogger()


def first_diff_in_strings(str1, str2):

    total_chars = min(len(str1), len(str2))
    i = 0

    while i < total_chars and str1[i] == str2[i]:
        i += 1

    if i == total_chars and len(str1) == len(str2):
        end_pos = None

    return end_pos


def assert_csv_diff(
    csv_file_one,
    csv_file_two,
    rows_to_skip_from_input=[],
    cols_to_skip_from_input=[]
):

    def assert_key_not_in_diff(diff, key):
        assert key not in diff, \
            "{}:\n{}".format(key, diff[key])

    def pop_items(diff_as_dict, rows, columns):
        for row in rows:
            for column in columns:
                key_to_pop = f"root[{str(row)}]['{column}']"
                assert key_to_pop in diff_as_dict, \
                    f"\"{key_to_pop}\" is expected to be in:\n{diff_as_dict}"
                diff_as_dict.pop(key_to_pop)

    csv_as_dict_one = read_csv_as_dict(csv_file_one)
    csv_as_dict_two = read_csv_as_dict(csv_file_two)
    diff = compute_dicts_diff(csv_as_dict_one, csv_as_dict_two)
    assert_key_not_in_diff(diff, "dictionary_item_added")
    assert_key_not_in_diff(diff, "dictionary_item_removed")
    if "values_changed" in diff:
        values_changed = diff["values_changed"]

        pop_items(values_changed, rows_to_skip_from_input, cols_to_skip_from_input)

        assert len(values_changed) == 0, \
            "Values_changed:\n{}".format(values_changed)


class DummyObject:
    def __init__(self):
        pass


def make_dummy_object():
    return DummyObject()
