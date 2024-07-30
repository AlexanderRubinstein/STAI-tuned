import os
import sys
import subprocess
import threading


# local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utility.utils import (
    NEW_SHELL_INIT_COMMAND,
    # raise_unknown,
    # check_dict,
    get_with_assert,
    # update_dict_by_nested_key,
    # pretty_json,
    # runcmd,
    log_or_print,
    error_or_print
)
# from utility.logger import (
#     log_or_print
# )
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
    # runcmd(cmd_to_run, logger=logger, verbose=False)
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
        conda_cmd = conda_cmd.replace("source", "sh")
    else:
        conda_cmd = ""

    if is_python:
        python_cmd = "python "
    else:
        python_cmd = ""

    # output_str = f" >> {logger.stdout_file} 2>>{logger.stderr_file}"

    return (
        conda_cmd
        + python_cmd
        + exec_path
        + kwargs_str
        # + output_str
    )


def run_cmd_through_popen(cmd_to_run, logger):

    # def read_stdout(process):
    #     for stdout_line in iter(process.stdout.readline, ''):
    #         print(f"STDOUT: {stdout_line}", end='')  # Process the line from stdout

    # def read_stderr(process):
    #     for stderr_line in iter(process.stderr.readline, ''):
    #         print(f"STDERR: {stderr_line}", end='')  # Process the line from stderr

    # def read_out(output, log_type):
    def read_out(output, log_func, logger):
        print("test read_out")
        # for stdout_line in iter(output.readline, ''):
        # while True:
            # for stdout_line in iter(lambda: output.read(1), b""):
            #     print("test")
            #     # if log_type == "error":
            #     #     # error_or_print(stdout_line, logger)
            #     #     # print(stdout_line, file=sys.stderr)
            #     #     # out = sys.stderr
            #     #     log_func = error_or_print
            #     # else:
            #     #     assert log_type == "info"
            #     #     # out = sys.stdout
            #     #     log_func = log_or_print
            #         # log_or_print(stdout_line, logger)
            #     # print(stdout_line, file=out)
            #     log_func(stdout_line, logger)
        for stdout_line in iter(lambda: output.read(1), b""):
            print("test")
            log_func(stdout_line, logger)

    # Start the process
    # process = subprocess.Popen(
    #     # ['your_command', 'arg1', 'arg2'],??
    #     cmd_to_run.split(),
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     # text=True,
    #     shell=True
    # )
    cmd_as_list = cmd_to_run.split()
    cmd_as_list = [
        "echo",
        "check test 1",
        "&&",
        '/usr/bin/sh',
        '/home/oh/arubinstein17/.bashrc',
        "&&",
        "echo",
        "check test",
        # '&&',
        # 'conda',
        # 'activate',
        # './envs/seg_env',
        # '&&',
        # 'python',
        # '/mnt/lustre/work/oh/arubinstein17/github/densification/src/densifier/proposal_generation.py',
        # '--config-file',
        # './src/densifier/get_segments/cropformer_hornet.yaml',
        # '--input',
        # './data/CounterAnimal/symlinked/common.tar.gz',
        # '--opts',
        # 'MODEL.WEIGHTS',
        # './models/CropFormer_hornet_3x/CropFormer_model/Entity_Segmentation/CropFormer_hornet_3x/CropFormer_hornet_3x_03823a.pth',
        # '--output',
        # './data/masks/common_masks_2907.pkl'
    ]
    process = subprocess.Popen(
        # ['your_command', 'arg1', 'arg2'],??
        cmd_as_list,
        stdout=sys.stdout,
        stderr=sys.stderr,
        # text=True,
        # shell=True
    )
    print(cmd_to_run.split()) # tmp

    log_or_print(f"Process started by runner has id: {process.pid}", logger)

    # # Start threads to read stdout and stderr
    # stdout_thread = threading.Thread(
    #     target=read_out,
    #     args=(process.stdout,log_or_print,logger)
    # )
    # stderr_thread = threading.Thread(
    #     target=read_out,
    #     args=(process.stderr,error_or_print,logger)
    # )

    # stdout_thread.start()
    # stderr_thread.start()

    # # sys.stderr = process.stderr
    # # sys.stdout = process.stdout # tmp

    # # # Wait for the process to complete
    # # process.wait()

    # # Wait for the threads to finish
    # stdout_thread.join()
    # stderr_thread.join()

    # Wait for the process to complete
    process.wait()

    if process.returncode != 0:
        raise Exception(
            f"Process failed with return code: {process.returncode}. "
            f"Check stderr for details"
        )
