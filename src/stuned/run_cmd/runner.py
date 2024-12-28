import os
import sys
# import subprocess
# import threading


# local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utility.utils import (
    NEW_SHELL_INIT_COMMAND,
    get_with_assert,
    # log_or_print,
    # error_or_print,
    run_cmd_through_popen
)
sys.path.pop(0)


# MAX_BUFFER_SIZE = 1000


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
        experiment_config.get("two_dash_flags"),
        experiment_config.get("single_dash_flags"),
        experiment_config.get("is_python", True),
        experiment_config.get("is_bash", False),
        get_with_assert(experiment_config, "conda_env"),
        logger=logger
    )

    logger.log(f"Running command:\n{cmd_to_run}", auto_newline=True)
    logger.log(f"As one line:\n{cmd_to_run}", auto_newline=False)
    run_cmd_through_popen(cmd_to_run, logger)


def make_task_cmd(
    exec_path,
    kwargs,
    two_dash_flags,
    single_dash_flags,
    is_python,
    is_bash,
    conda_env,
    logger
):

    if kwargs is not None:
        kwargs_str = " " + " ".join([
            "--{}={}".format(k, v)
            for k, v in kwargs.items()
        ])
    else:
        kwargs_str = ""

    if two_dash_flags is not None:
        two_dash_flags_str = " " + " ".join([
            "--{}".format(arg)
            for arg in two_dash_flags
        ])
    else:
        two_dash_flags_str = ""

    if single_dash_flags is not None:
        single_dash_flags_str = " " + " ".join([
            "-{}".format(arg)
            for arg in single_dash_flags
        ])
    else:
        single_dash_flags_str = ""

    if conda_env is not None:
        # assert is_python
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

    if is_bash:
        bash_cmd = "bash "
    else:
        bash_cmd = ""

    return (
        conda_cmd
        + python_cmd
        + bash_cmd
        + exec_path
        + kwargs_str
        + two_dash_flags_str
        + single_dash_flags_str
    )
