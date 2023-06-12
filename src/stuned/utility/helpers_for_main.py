import argparse


# local modules
from .utils import (
    apply_random_seed,
    pretty_json,
    kill_processes
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an experiment with given configs."
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="path to config file")
    return parser.parse_args()


def prepare_wrapper_for_experiment(check_config):

    def wrapper_for_experiment(run_experiment):

        def run_experiment_with_logger():

            logger = make_logger_with_tmp_output_folder()
            processes_to_kill_before_exiting = []

            try:

                main_args = parse_args()

                experiment_config = get_config(
                    main_args.config_path,
                    check_config,
                    logger
                )

                with redneck_logger_context(
                    experiment_config[LOGGING_CONFIG_KEY],
                    experiment_config["current_run_folder"],
                    logger=logger,
                    exp_name=experiment_config[EXP_NAME_CONFIG_KEY],
                    start_time=None,
                    config_to_log_in_wandb=experiment_config
                ) as logger:

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
