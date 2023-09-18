import random
import sys
from time import sleep

import torch
import wandb

from stuned.utility.helpers_for_main import prepare_wrapper_for_experiment
from stuned.utility.logger import (
    try_to_log_in_wandb,
    try_to_log_in_csv, try_to_sync_csv_with_remote
)
def check_config_for_demo_experiment(config, config_path, logger):
    assert "initialization_type" in config
    assert "image" in config


def demo_experiment(
    experiment_config,
    logger,
    processes_to_kill_before_exiting
):
    image_size = experiment_config["image"]["shape"]

    colored_image = torch.zeros((3, *image_size))

    if experiment_config["image"]["color"] == "red":
        channel = 0
    else:
        channel = 2

    init_type = experiment_config["initialization_type"]
    # put random seed based on the timestamp of the experiment unix time
    # random.seed(datetime.now().timestamp())
    if init_type == "2" or init_type == 2:
        sleep_time = random.randint(7, 15)
    else:
        sleep_time = random.randint(1, 4)

    try_to_log_in_csv(logger, "color_required", experiment_config["image"]["color"])
    for i in range(10):
        if init_type == "random":
            colored_image[channel] = torch.randn(image_size)
        else:
            colored_image[channel] = torch.ones(image_size)

        mean = colored_image.mean()

        wandb_stats_to_log = {
            "Confusing sample before optimization":
            wandb.Image(
                colored_image,
                caption=
                    f"Initialization type: {init_type}"
            ),
            "mean": mean
        }

        # log image + mean in wandb
        try_to_log_in_wandb(
            logger,
            wandb_stats_to_log,
            step=i
        )

        # log like 20 metrics to csv
        for j in range(10):
            try_to_log_in_csv(
                logger,
                f"metric_{j}",
                random.randint(0, 100)
            )

        # log latest mean in csv
        # try_to_log_in_csv(logger, "mean of latest tensor", mean + 3)

        sleep(sleep_time)
        # try_to_sync_csv_with_remote(logger)

        colored_image[:, :, :] = 0


def main(config):
    # python fire library
    # try:
    #     import pydevd_pycharm
    #     pydevd_pycharm.settrace('localhost', port=11112, stdoutToServer=True, stderrToServer=True)
    # except:
    #     pass
    # logger = job_manager.begin(config=config)
    prepare_wrapper_for_experiment()(
        demo_experiment
    )()

if __name__ == "__main__":
    main(sys.argv[1:])
