import argparse
import git
import gspread

# local modules
from .utils import (
    apply_random_seed,
    pretty_json,
    kill_processes,
    get_project_root_path
)
from .configs import (
    get_config
)
from .constants import EXP_NAME_CONFIG_KEY
from .logger import (
    LOGGING_CONFIG_KEY,
    make_logger_with_tmp_output_folder,
    handle_exception,
    redneck_logger_context
)

class BestPerformanceTracker():
    """
    Given a performance metric, this class tracks the best performance together with the best `step`
    at which the best performance was achieved. This is useful when recording the perofrmances to gsheets
    """

    def __init__(self):
        self.performances = {}

    def update(self, performance_key, performance_val, step, higher_is_better=True):
        if performance_key not in self.performances:
            self.performances[performance_key] = {
                "best_performance": performance_val,
                "best_step": step,
                "updated": True
            }
        else:
            if (
                (higher_is_better and performance_val > self.performances[performance_key]["best_performance"])
                or (not higher_is_better and performance_val < self.performances[performance_key]["best_performance"])
            ):
                self.performances[performance_key]["best_performance"] = performance_val
                self.performances[performance_key]["best_step"] = step
                self.performances[performance_key]["updated"] = True

    def get_new_vals_to_log(self):
        vals_to_log = {}
        for performance_key in self.performances:
            if self.performances[performance_key]["updated"]:
                vals_to_log["best"+str(performance_key)] = self.performances[performance_key]["best_performance"]
                self.performances[performance_key]["updated"] = False
        return vals_to_log
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

                main_args = parse_args()

                config_path = main_args.config_path

                experiment_config = get_config(
                    config_path,
                    logger
                )


                using_socket = False
                # Check if socket is being used
                if "logging" in experiment_config and "server_ip" in experiment_config["logging"] and "server_port" in experiment_config["logging"]:
                    logger.log("Using socket for logging")
                    using_socket = True
                if patch_config is not None:
                    patch_config(experiment_config)

                with redneck_logger_context(
                    experiment_config[LOGGING_CONFIG_KEY],
                    experiment_config["current_run_folder"],
                    logger=logger,
                    exp_name=experiment_config[EXP_NAME_CONFIG_KEY],
                    start_time=None,
                    config_to_log_in_wandb=experiment_config,
                    using_socket = using_socket
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
                    if "params" in experiment_config and "random_seed" in experiment_config["params"]:
                        apply_random_seed(
                            experiment_config["params"]["random_seed"]
                        )
                    else:
                        logger.log("Warning: no random seed was found in the config file. Not setting the seed.")

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


def get_rows_from_gsheet(worksheet, row_ids: list) -> list:
    """
    Fetches rows from a Google Sheet based on the provided row IDs.

    :param worksheet: The gspread worksheet object.
    :param row_ids: A list of row numbers to fetch.
    :return: A list of lists where each inner list represents a row's values.
    """

    # Since we don't know the last column, let's use the maximum possible column letter 'ZZZ'
    # (this is a workaround since gspread doesn't provide a direct way to fetch entire rows without specifying the end column)
    ranges = [f'A{row_id}:ZZZ{row_id}' for row_id in row_ids]

    # Fetch the rows with a single API call
    rows_data = worksheet.batch_get(ranges)

    rows_data = [row[0] for row in rows_data]

    return rows_data

def get_key_vals_of_row(worksheet, row_id):
    """
    Fetches the key-value pairs of a row from a Google Sheet.

    :param worksheet: The gspread worksheet object.
    :param row_id: The row number to fetch.
    :return: A dictionary of key-value pairs.
    """

    # Fetch the row
    row_data = get_rows_from_gsheet(worksheet, [1, row_id])

    # Convert the row to a dictionary
    key_vals = dict(zip(row_data[0], row_data[1]))

    return key_vals