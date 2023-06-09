import os
import sys
import torch
import wandb


from stuned.utility.helpers_for_main import prepare_wrapper_for_experiment
from stuned.utility.logger import (
    try_to_log_in_wandb,
    try_to_log_in_csv
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

        # log latest mean in csv
        try_to_log_in_csv(logger, "mean of latest tensor", mean)

        colored_image[:, :, :] = 0


def main():
    prepare_wrapper_for_experiment(check_config_for_demo_experiment)(
        demo_experiment
    )()


if __name__ == "__main__":
    main()
