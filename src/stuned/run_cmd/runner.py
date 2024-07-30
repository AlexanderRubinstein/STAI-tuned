import os
import sys
import subprocess
import threading


# local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utility.utils import (
    NEW_SHELL_INIT_COMMAND,
    get_with_assert,
    log_or_print,
    error_or_print
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
    run_cmd_through_popen(cmd_to_run, logger)


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

    return (
        conda_cmd
        + python_cmd
        + exec_path
        + kwargs_str
    )


def run_cmd_through_popen(cmd_to_run, logger):

    def read_out(process, out_type, logger):
        buffer = ""
        if out_type == "stdout":
            out = process.stdout
            log_func = log_or_print
        else:
            assert out_type == "stderr"
            out = process.stderr
            log_func = error_or_print

        while True:
            output = out.readline()
            if output == '' and process.poll() is not None:
                break

            buffer += output
            if output == '\n':
                log_func(buffer, logger)
                buffer = ""

        if buffer != "":
            log_func(buffer, logger)

    process = subprocess.Popen(
        cmd_to_run,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    log_or_print(f"Process started by runner has id: {process.pid}", logger)

    stdout_thread = threading.Thread(
        target=read_out,
        args=(process,"stdout",logger)
    )
    stderr_thread = threading.Thread(
        target=read_out,
        args=(process,"stderr",logger)
    )

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    process.wait()

    if process.returncode != 0:
        raise Exception(
            f"Process failed with return code: {process.returncode}. "
            f"Check stderr for details"
        )

