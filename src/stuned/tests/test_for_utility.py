import sys
import os
import re
from tempfile import TemporaryDirectory

# local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utility.helpers_for_tests import (
    wrapper_for_test,
    make_dummy_logger,
    assert_csv_diff
)
from utility.utils import (
    MAX_RETRIES_ERROR_MSG,
    retrier_factory,
    get_project_root_path,
    remove_file_or_folder,
    expand_csv,
    touch_file,
    folder_still_has_updates
)
from utility.logger import (
    retrier_factory_with_auto_logger
)


FAIL_RETURN = -42
SUCCESS_RETURN = 42
assert FAIL_RETURN != SUCCESS_RETURN
SUCCESS_LOG = "All is good, finally!"
MAX_RETRIES = 5
SLEEP_TIME = 1
CODE_LINE_PLACEHOLDER = "<LINE>"
PATH_PLACEHOLDER = "<PATH>"
EXCEPTION_TO_RERAISE = "Active exception to reraise"


def final_func(logger):
    global global_n_fails
    global_n_fails = 0
    logger.error(MAX_RETRIES_ERROR_MSG)
    return FAIL_RETURN


def make_error_log(current_retry, total_retries):
    return (
        "Traceback (most recent call last):\n"
        "  File \"{}/utils.py\", line {}, in wrapped_func\n"
        "    return func(*args, **kwargs)\n"
        "  File \"{}/test_for_utility.py\", line {}, in fail_n\n"
        "    raise Exception(f\"Fail {{test_cache[\'global_n_fails\']}}\")\n"
        "Exception: Fail {}"
        "\n\nRetrying fail_n: {}/{}.\n\n"
    ).format(
        PATH_PLACEHOLDER,
        CODE_LINE_PLACEHOLDER,
        PATH_PLACEHOLDER,
        CODE_LINE_PLACEHOLDER,
        current_retry,
        current_retry,
        total_retries
    )


@wrapper_for_test
def test_retrier():

    test_cache = {
        'global_n_fails': 0,
        'dummy_logger': make_dummy_logger()
    }

    @retrier_factory(
        test_cache['dummy_logger'],
        final_func,
        max_retries=MAX_RETRIES,
        sleep_time=SLEEP_TIME
    )
    def fail_n(n, test_cache):
        if test_cache['global_n_fails'] < n:
            test_cache['global_n_fails'] += 1
            raise Exception(f"Fail {test_cache['global_n_fails']}")
        test_cache["dummy_logger"].log(SUCCESS_LOG)
        return SUCCESS_RETURN

    def make_total_error_log(n, total_retries=MAX_RETRIES):
        res = ""
        for i in range(1, n + 1):
            res += make_error_log(i, total_retries)
        return res

    def reset(test_cache):
        test_cache["dummy_logger"].reset()
        test_cache["global_n_fails"] = 0

    def change_for_placeholders(input_string):
        res = re.sub(
            r"line \d*",
            f"line {CODE_LINE_PLACEHOLDER}",
            input_string,
            count=0
        )
        res = re.sub(
            r"\".*/utils.py\"",
            f"\"{PATH_PLACEHOLDER}/utils.py\"",
            res,
            count=0
        )
        res = re.sub(
            r"\".*/test_for_utility.py\"",
            f"\"{PATH_PLACEHOLDER}/test_for_utility.py\"",
            res,
            count=0
        )
        return res

    def func_with_one_arg(some_arg):
        if some_arg == FAIL_RETURN:
            raise Exception(EXCEPTION_TO_RERAISE)
        else:
            return some_arg

    def func_with_nested_retrier(some_arg, logger):

        @retrier_factory(logger, sleep_time=0)
        def func_with_given_logger(some_arg):
            return func_with_one_arg(some_arg)

        return func_with_given_logger(some_arg)

    @retrier_factory_with_auto_logger(sleep_time=0)
    def func_with_inferred_logger(some_arg, logger):
        return func_with_one_arg(some_arg)

    def assert_error_logs(test_cache, expected_error_logs):
        assert (
            change_for_placeholders(test_cache["dummy_logger"].error_strings)
                == expected_error_logs
        )
        reset(test_cache)

    def test_failing(n_fails, return_code, log_strings, test_cache):
        res = fail_n(n_fails, test_cache)
        expected_error = make_total_error_log(min(n_fails, MAX_RETRIES))
        if n_fails > MAX_RETRIES:
            expected_error += MAX_RETRIES_ERROR_MSG
        assert res == return_code
        assert test_cache["dummy_logger"].log_strings == log_strings
        assert_error_logs(test_cache, expected_error)

    def test_nested_retrier(func, test_cache):
        assert (
            func(SUCCESS_RETURN, test_cache["dummy_logger"])
                == SUCCESS_RETURN
        )
        assert len(test_cache["dummy_logger"].error_strings) == 0
        reset(test_cache)
        try:
            func(FAIL_RETURN, test_cache["dummy_logger"])
        except Exception as e:
            assert str(e) == EXCEPTION_TO_RERAISE
        assert (
            len(test_cache["dummy_logger"].error_strings)
                > len(MAX_RETRIES_ERROR_MSG)
        )
        assert (
            test_cache["dummy_logger"].error_strings[-len(MAX_RETRIES_ERROR_MSG):]
                == MAX_RETRIES_ERROR_MSG
        )
        reset(test_cache)

    # success without retries
    test_failing(0, SUCCESS_RETURN, SUCCESS_LOG, test_cache)

    # fail some retries
    test_failing(3, SUCCESS_RETURN, SUCCESS_LOG, test_cache)

    # fail each retry
    test_failing(6, FAIL_RETURN, "", test_cache)

    # nested retrier
    test_nested_retrier(func_with_nested_retrier, test_cache)
    test_nested_retrier(func_with_inferred_logger, test_cache)


@wrapper_for_test
def test_expanding_csvs():
    test_data_folder = os.path.join(
        get_project_root_path(),
        "src",
        "stuned",
        "tests",
        "data"
    )
    csv_to_expand = os.path.join(test_data_folder, "csv_to_expand.csv")
    expanded_csv = os.path.join(test_data_folder, "expanded_csv.csv")

    tmp_csv = os.path.join(test_data_folder, "tmp_csv.csv")
    expand_csv(csv_to_expand, tmp_csv)
    assert_csv_diff(
        tmp_csv,
        expanded_csv
    )
    remove_file_or_folder(tmp_csv)


@wrapper_for_test
def test_filewatcher():

    with TemporaryDirectory() as dir_to_watch:
        file1_path = os.path.join(dir_to_watch, "file1")
        touch_file(file1_path)

        # delta > max_time
        check_time = 0.1
        has_events_bool = folder_still_has_updates(
            dir_to_watch,
            2,
            1,
            check_time=check_time
        )
        assert has_events_bool

        print("lol", file=open(file1_path, 'w'))
        # delta < max_time
        has_events_bool = folder_still_has_updates(
            dir_to_watch,
            1,
            2,
            check_time=check_time
        )
        assert not has_events_bool


def main():
    test_retrier()
    test_expanding_csvs()
    test_filewatcher()


if __name__ == "__main__":
    main()
