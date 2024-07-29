import argparse
import git
import os


# local modules
from .utils import (
    apply_random_seed,
    pretty_json,
    kill_processes,
    get_project_root_path,
    find_by_subkey
)
from .configs import (
    EXP_NAME_CONFIG_KEY,
    get_config
)
from .logger import (
    LOGGING_CONFIG_KEY,
    make_logger_with_tmp_output_folder,
    handle_exception,
    redneck_logger_context
)


SCRATCH_LOCAL = os.path.join(os.path.abspath(os.sep), "scratch_local")
SCRATCH_VAR_NAME = "SCRATCH"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an experiment with given configs."
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="path to config file")
    return parser.parse_args()


def prepare_wrapper_for_experiment(check_config=None, patch_config=None):

    def wrapper_for_experiment(run_experiment):

        def run_experiment_with_logger():

            logger = make_logger_with_tmp_output_folder()
            processes_to_kill_before_exiting = []

            try:

                define_env_vars()

                main_args = parse_args()

                config_path = main_args.config_path

                experiment_config = get_config(
                    config_path,
                    logger
                )

                if patch_config is not None:
                    patch_config(experiment_config)

                with redneck_logger_context(
                    # experiment_config[LOGGING_CONFIG_KEY],
                    experiment_config.get(LOGGING_CONFIG_KEY, {}),
                    experiment_config["current_run_folder"],
                    logger=logger,
                    exp_name=experiment_config[EXP_NAME_CONFIG_KEY],
                    start_time=None,
                    config_to_log_in_wandb=experiment_config
                ) as logger:

                    repo = git.Repo(get_project_root_path())
                    sha = repo.head.object.hexsha
                    logger.log(f"Hash of current git commit: {sha}")

                    if check_config is not None:
                        logger.log(
                            "Checking config: {}".format(
                                config_path
                                    if config_path
                                    else "HARDCODED_CONFIG in utility/configs.py"
                            ),
                            auto_newline=True
                        )
                        check_config(experiment_config, config_path, logger=logger)

                    logger.log(
                        "Experiment config:\n{}".format(
                            pretty_json(experiment_config)
                        )
                    )

                    apply_random_seed(
                        experiment_config["params"]["random_seed"]
                    )

                    run_experiment(
                        experiment_config,
                        logger,
                        processes_to_kill_before_exiting
                    )

            except Exception as e:
                handle_exception(logger, e)
            except:
                handle_exception(logger)
            finally:
                kill_processes(processes_to_kill_before_exiting)

        return run_experiment_with_logger

    return wrapper_for_experiment


def define_env_vars():
    if SCRATCH_VAR_NAME not in os.environ and os.path.exists(SCRATCH_LOCAL):
        user_name = os.environ.get("USER")
        any_folder_of_user = find_by_subkey(
            os.listdir(SCRATCH_LOCAL),
            user_name,
            assert_found=True,
            only_first_occurence=True
        )
        os.environ[SCRATCH_VAR_NAME] = os.path.join(
            SCRATCH_LOCAL,
            any_folder_of_user
        )
