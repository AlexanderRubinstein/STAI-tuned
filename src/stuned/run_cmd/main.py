import os
import sys


# local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from run_cmd.runner import (
    runner,
    check_runner_config,
    patch_runner_config
)
from utility.helpers_for_main import prepare_wrapper_for_experiment
sys.path.pop(0)


def main():
    prepare_wrapper_for_experiment(
        check_runner_config,
        patch_runner_config
    )(
        runner
    )()


if __name__ == "__main__":
    main()
