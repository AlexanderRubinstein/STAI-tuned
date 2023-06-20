import argparse
import os
import copy
import contextlib
from tempfile import NamedTemporaryFile
import subprocess
import shutil
import sys


# local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utility.utils import (
    DEFAULT_ENV_NAME,
    NEW_SHELL_INIT_COMMAND,
    make_autogenerated_config_name,
    read_yaml,
    save_as_yaml,
    update_dict_by_nested_key,
    get_project_root_path,
    decode_strings_in_dict,
    read_csv_as_dict,
    normalize_path,
    check_duplicates,
    itself_and_lower_upper_case,
    expand_csv,
    retrier_factory
)
from utility.configs import (
    AUTOGEN_PREFIX,
    NESTED_CONFIG_KEY_SEPARATOR,
    make_csv_config
)
from utility.logger import (
    STATUS_CSV_COLUMN,
    SUBMITTED_STATUS,
    WHETHER_TO_RUN_COLUMN,
    DELTA_PREFIX,
    SLURM_PREFIX,
    PREFIX_SEPARATOR,
    make_logger,
    make_gdrive_client,
    sync_local_file_with_gdrive,
    log_csv_for_concurrent,
    make_progress_bar,
    fetch_csv,
    try_to_upload_csv
)


FILES_URL = "https://drive.google.com/file"


USE_SRUN = False


PATH_TO_DEFAULT_CONFIG_COLUMN = "path_to_default_config"
MAIN_PATH_COLUMN = "path_to_main"
DEV_NULL = "/dev/null"
DEFAULT_SLURM_ARGS_DICT = {
    "partition": "gpu-2080ti-beegfs",
    "gpus": 1,
    "time": "02:00:00",
    "ntasks": 1,
    "cpus-per-task": 2,
    "error": DEV_NULL,
    "output": DEV_NULL
}
EMPTY_STRING = "EMPTY_STRING"
PLACEHOLDERS_FOR_DEFAULT = itself_and_lower_upper_case("Default")
EXPANDED_CSV_PREFIX = "expanded_"
TO_RUN = "to_run"
TO_SKIP = "to_skip"
assert TO_RUN != TO_SKIP
CURRENT_ROW_PLACEHOLDER = "__ROW__"
CURRENT_WORKSHEET_PLACEHOLDER = "__WORKSHEET__"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and/or validate models."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="path to csv file"
    )
    parser.add_argument(
        "--conda_env",
        type=str,
        required=False,
        default=DEFAULT_ENV_NAME,
        help="conda environment name"
    )
    parser.add_argument(
        "--run_locally",
        action="store_true",
        help="whether to run this script locally"
    )
    parser.add_argument(
        "--log_file_path",
        type=str,
        required=False,
        default=get_default_log_file_path(),
        help="default path for the log file"
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="whether to first expand input csv by cartesian product of options"
    )
    return parser.parse_args()


def get_default_log_file_path():
    return os.path.join(
        get_project_root_path(),
        "tmp",
        "tmp_log_for_run_from_csv.out"
    )


def main():
    args = parse_args()

    logger = make_logger()

    csv_path_or_url = args.csv_path

    logger.log(f"Fetching csv from: {csv_path_or_url}")
    csv_path, spreadsheet_url, worksheet_name, gspread_client = fetch_csv(
        csv_path_or_url,
        logger
    )

    if args.expand:
        expand_gsheet(csv_path, spreadsheet_url, worksheet_name, gspread_client)

    inputs_csv = read_csv_as_dict(csv_path)
    rows_to_run = []
    progress_bar = make_progress_bar(
        len(inputs_csv),
        f"Processing rows.",
        logger
    )
    for row_number, csv_row in inputs_csv.items():
        whether_to_run, run_cmd = process_csv_row(
            csv_row,
            row_number,
            csv_path,
            args.conda_env,
            args.run_locally,
            args.log_file_path,
            spreadsheet_url,
            worksheet_name,
            logger
        )
        if whether_to_run == TO_RUN:
            rows_to_run.append((row_number, run_cmd))
        progress_bar.update()
    concurrent_log_func = retrier_factory()(log_csv_for_concurrent)

    csv_updates = []
    for row_number, _ in rows_to_run:
        csv_updates.append((row_number, STATUS_CSV_COLUMN, SUBMITTED_STATUS))
        csv_updates.append((row_number, WHETHER_TO_RUN_COLUMN, "0"))

    concurrent_log_func(
        csv_path,
        csv_updates
    )

    os.makedirs(os.path.dirname(args.log_file_path), exist_ok=True)
    with open(args.log_file_path, 'w+') as log_file:
        for row_number, run_cmd in rows_to_run:

            subprocess.call(
                run_cmd,
                stdout=log_file,
                stderr=log_file,
                shell=True
            )

    try_to_upload_csv(
        csv_path,
        spreadsheet_url,
        worksheet_name,
        gspread_client
    )


def expand_gsheet(csv_path, spreadsheet_url, worksheet_name, gspread_client):

    expanded_csv_path = os.path.join(
        os.path.dirname(csv_path),
        EXPANDED_CSV_PREFIX + os.path.basename(csv_path)
    )
    expand_csv(
        csv_path,
        expanded_csv_path
    )
    csv_path = expanded_csv_path
    if worksheet_name is not None:
        worksheet_name = EXPANDED_CSV_PREFIX + worksheet_name

    try_to_upload_csv(
        csv_path,
        spreadsheet_url,
        worksheet_name,
        gspread_client
    )


def fetch_default_config_path(path, logger):

    if FILES_URL in path:

        gdrive_client = make_gdrive_client(logger)
        with (
            NamedTemporaryFile('w+t', delete=False) as tmp_file
        ):
            remote_file = sync_local_file_with_gdrive(
                gdrive_client,
                tmp_file.name,
                path,
                download=True,
                logger=logger
            )
            file_path = os.path.join(
                get_default_configs_folder(),
                remote_file["title"].split('.')[0],
                "config.yaml"
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            shutil.move(tmp_file.name, file_path)
            return file_path

    else:
        return normalize_path(path)


def get_default_configs_folder():
    return os.path.join(
        get_project_root_path(),
        "experiment_configs"
    )


def process_csv_row(
    csv_row,
    row_number,
    input_csv_path,
    conda_env,
    run_locally,
    log_file_path,
    spreadsheet_url,
    worksheet_name,
    logger
):

    def check_csv_row(csv_row):

        assert MAIN_PATH_COLUMN in csv_row

        assert WHETHER_TO_RUN_COLUMN in csv_row

        assert PATH_TO_DEFAULT_CONFIG_COLUMN in csv_row

        for i, key in enumerate(csv_row.keys()):
            assert key is not None, \
                f"Column {i} has empty column name. " \
                f"Or some table entries contain commas."
            if PREFIX_SEPARATOR in key:
                assert (
                        DELTA_PREFIX in key
                    or
                        SLURM_PREFIX in key
                ), \
                f"Did not find \"{DELTA_PREFIX}\" " \
                f"or \"{SLURM_PREFIX}\" in \"{key}\""

    check_csv_row(csv_row)

    replace_placeholders(csv_row, CURRENT_ROW_PLACEHOLDER, str(row_number))
    replace_placeholders(csv_row, CURRENT_WORKSHEET_PLACEHOLDER, worksheet_name)

    if (
        not csv_row[WHETHER_TO_RUN_COLUMN].isnumeric()
            or int(csv_row[WHETHER_TO_RUN_COLUMN]) == 0
    ):
        return TO_SKIP, None

    default_config_path = fetch_default_config_path(
        csv_row[PATH_TO_DEFAULT_CONFIG_COLUMN],
        logger
    )

    assert os.path.exists(default_config_path)

    exp_dir = normalize_path(os.path.dirname(default_config_path))
    exp_name = os.path.basename(exp_dir)

    default_config = read_yaml(default_config_path)

    _, new_config_path = make_new_config(
        csv_row,
        row_number,
        input_csv_path,
        default_config,
        exp_dir,
        spreadsheet_url,
        worksheet_name
    )

    cmd_as_string = make_task_cmd(
        new_config_path,
        conda_env,
        normalize_path(csv_row[MAIN_PATH_COLUMN])
    )

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    with (
        NamedTemporaryFile('w', delete=False)
            if not (run_locally or USE_SRUN)
            else contextlib.nullcontext()
    ) as tmp_file:
        if run_locally:
            final_cmd = "{} &> {} &".format(cmd_as_string, log_file_path)
        else:
            slurm_args_dict = make_slurm_args_dict(
                csv_row,
                exp_name,
                log_file_path
            )
            if USE_SRUN:
                slurm_args_as_string = " ".join(
                    [
                        f"--{flag}={value}"
                            for flag, value
                                in slurm_args_dict.items()
                    ]
                )
                final_cmd = "srun {} sh -c \"{}\" &".format(
                    slurm_args_as_string,
                    cmd_as_string
                )
            else:
                fill_sbatch_script(tmp_file, slurm_args_dict, cmd_as_string)
                final_cmd = "sbatch {}".format(tmp_file.name)

    return TO_RUN, final_cmd


def fill_sbatch_script(sbatch_file, slurm_args_dict, command):

    sbatch_file.write("#!/bin/bash\n")

    for slurm_arg, value in slurm_args_dict.items():
        sbatch_file.write("#SBATCH --{}={}\n".format(slurm_arg, value))

    sbatch_file.write(command)
    sbatch_file.flush()


def make_new_config(
    csv_row,
    row_number,
    input_csv_path,
    default_config,
    exp_dir,
    spreadsheet_url,
    worksheet_name
):

    deltas = extract_from_csv_row_by_prefix(
        csv_row,
        DELTA_PREFIX + PREFIX_SEPARATOR
    )

    if len(deltas) > 0:
        check_duplicates(list(deltas.keys()))

    keys_to_pop = []
    for key, value in deltas.items():
        if value == EMPTY_STRING:
            deltas[key] = ""
        if value in PLACEHOLDERS_FOR_DEFAULT:
            keys_to_pop.append(key)
        elif value == "":
            raise Exception("Empty value for delta:{}".format(key))

    for key in keys_to_pop:
        deltas.pop(key)

    decode_strings_in_dict(
        deltas,
        list_separators=[' '],
        list_start_symbol='[',
        list_end_symbol=']'
    )

    deltas["logging/output_csv"] = make_csv_config(
        input_csv_path,
        row_number,
        spreadsheet_url,
        worksheet_name
    )
    new_config = make_config_from_default_and_deltas(default_config, deltas)
    new_config_path = os.path.join(
        exp_dir,
        AUTOGEN_PREFIX,
        make_autogenerated_config_name(input_csv_path, row_number)
    )
    os.makedirs(os.path.dirname(new_config_path), exist_ok=True)
    save_as_yaml(
        new_config_path,
        new_config
    )
    return new_config, new_config_path


def replace_placeholders(csv_row, placeholder, new_value):
    for column_name, value in csv_row.items():
        csv_row[column_name] = value.replace(placeholder, new_value)


def make_task_cmd(new_config_path, conda_env, exec_path):
    exec_args = "--config_path {}".format(new_config_path)
    return "{} {} && python {} {}".format(
        NEW_SHELL_INIT_COMMAND,
        conda_env,
        exec_path,
        exec_args
    )


def make_slurm_args_dict(csv_row, exp_name, log_file):

    specified_slurm_args = extract_from_csv_row_by_prefix(
        csv_row,
        SLURM_PREFIX + PREFIX_SEPARATOR
    )

    all_slurm_args_dict = copy.deepcopy(DEFAULT_SLURM_ARGS_DICT)

    all_slurm_args_dict["job-name"] = exp_name

    all_slurm_args_dict["output"] = log_file
    all_slurm_args_dict["error"] = log_file

    for flag, value in specified_slurm_args.items():
        if not value in PLACEHOLDERS_FOR_DEFAULT:
            # override slurm args if given
            all_slurm_args_dict[flag] = value

    os.makedirs(os.path.dirname(all_slurm_args_dict["output"]), exist_ok=True)
    os.makedirs(os.path.dirname(all_slurm_args_dict["error"]), exist_ok=True)

    return all_slurm_args_dict


def extract_from_csv_row_by_prefix(csv_row, prefix):
    prefix_len = len(prefix)
    result = {}
    for key, value in csv_row.items():
        assert key is not None, "Possibly inconsistent number of delimeters."
        if key == prefix:
            raise Exception(
                f"Found \"{prefix}\" (nothing after this prefix) "
                f"in csv_row:\n{csv_row}"
            )
        if len(key) > prefix_len and prefix == key[:prefix_len]:
            result[key[prefix_len:]] = value
    return result


def make_config_from_default_and_deltas(default_config, deltas):
    assert isinstance(deltas, dict)
    new_config = copy.deepcopy(default_config)
    for nested_config_key, new_value in deltas.items():
        update_dict_by_nested_key(
            new_config,
            nested_config_key.split(NESTED_CONFIG_KEY_SEPARATOR),
            new_value,
            to_create_new_elements=True
        )
    return new_config


if __name__ == "__main__":
    main()
