import os
import sys


# local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utility.utils import (
    NEW_SHELL_INIT_COMMAND,
    raise_unknown,
    check_dict,
    get_with_assert,
    update_dict_by_nested_key,
    pretty_json,
    runcmd
)
sys.path.pop(0)


def patch_runner_config(experiment_config):
    pass


def check_runner_config(experiment_config, config_path, logger=None):
    pass


def runner(
    experiment_config,
    logger=None,
    processes_to_kill_before_exiting=[]
):

    cmd_to_run = make_task_cmd(
        get_with_assert(experiment_config, "exec_path"),
        experiment_config.get("kwargs"),
        experiment_config.get("is_python", True),
        get_with_assert(experiment_config, "conda_env"),
        logger=logger
    )

    logger.log(f"Running command:\n{cmd_to_run}", auto_newline=True)
    logger.log(f"As one line:\n{cmd_to_run}", auto_newline=False)
    runcmd(cmd_to_run, logger=logger, verbose=False)


def make_task_cmd(exec_path, kwargs, is_python, conda_env, logger):

    if kwargs is not None:
        kwargs_str = " " + " ".join([
            "--{} {}".format(k, v)
            for k, v in kwargs.items()
        ])
    else:
        kwargs_str = ""

    if conda_env is not None:
        assert is_python
        conda_cmd = "{} {} && ".format(
            NEW_SHELL_INIT_COMMAND,
            conda_env
        )
    else:
        conda_cmd = ""

    if is_python:
        python_cmd = "python "
    else:
        python_cmd = ""

    output_str = f" >> {logger.stdout_file} 2>>{logger.stderr_file}"

    return (
        conda_cmd
        + python_cmd
        + exec_path
        + kwargs_str
        + output_str
    )
