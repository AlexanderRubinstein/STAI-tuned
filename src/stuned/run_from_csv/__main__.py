import argparse
import os
import copy
from tempfile import NamedTemporaryFile
import subprocess
import shutil
import sys
import multiprocessing as mp


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
    PLACEHOLDERS_FOR_DEFAULT,
    make_logger,
    make_gdrive_client,
    sync_local_file_with_gdrive,
    log_csv_for_concurrent,
    fetch_csv,
    try_to_upload_csv,
    make_delta_column_name
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
EXPANDED_CSV_PREFIX = "expanded_"
CURRENT_ROW_PLACEHOLDER = "__ROW__"
CURRENT_WORKSHEET_PLACEHOLDER = "__WORKSHEET__"
MAX_PROCESSES = 16


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


def main(make_final_cmd=None, allowed_prefixes=(SLURM_PREFIX, DELTA_PREFIX)):

    if make_final_cmd is None:
        make_final_cmd = make_final_cmd_slurm

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

    with mp.Manager() as shared_memory_manager:

        lock = shared_memory_manager.Lock()
        current_step = shared_memory_manager.Value("int", 0)
        shared_rows_to_run = shared_memory_manager.list()
        shared_csv_updates = shared_memory_manager.list()
        shared_default_config_paths = shared_memory_manager.dict()

        starmap_args_for_row_processing = [
            (
                make_final_cmd,
                csv_row,
                row_number,
                csv_path,
                args.conda_env,
                args.run_locally,
                args.log_file_path,
                spreadsheet_url,
                worksheet_name,
                logger,
                lock,
                shared_rows_to_run,
                shared_default_config_paths,
                shared_csv_updates,
                current_step,
                len(inputs_csv)
            )
                for row_number, csv_row in inputs_csv.items()
        ]

        if len(starmap_args_for_row_processing):

            first_csv_row = starmap_args_for_row_processing[0][1]
            check_csv_column_names(first_csv_row, allowed_prefixes)

            pool_size = get_pool_size(len(starmap_args_for_row_processing))

            upload_csv = False

            with mp.Pool(pool_size) as pool:

                pool.starmap(
                    process_csv_row,
                    starmap_args_for_row_processing
                )

                if len(shared_rows_to_run):

                    assert 2 * len(shared_rows_to_run) == len(shared_csv_updates)

                    concurrent_log_func = retrier_factory()(log_csv_for_concurrent)

                    concurrent_log_func(
                        csv_path,
                        shared_csv_updates
                    )

                    os.makedirs(os.path.dirname(args.log_file_path), exist_ok=True)

                    starmap_args_for_job_submitting = [
                        (
                            run_cmd,
                            args.log_file_path
                        )
                            for run_cmd in shared_rows_to_run
                    ]

                    pool.starmap(
                        submit_job,
                        starmap_args_for_job_submitting
                    )
                    upload_csv = True

            if upload_csv:
                try_to_upload_csv(
                    csv_path,
                    spreadsheet_url,
                    worksheet_name,
                    gspread_client
                )


def get_pool_size(iterable_len):
    return min(
        min(
            max(1, mp.cpu_count() - 1),
            iterable_len
        ),
        MAX_PROCESSES
    )


def submit_job(run_cmd, log_file_path):
    with open(log_file_path, 'w+') as log_file:
        subprocess.call(
            run_cmd,
            stdout=log_file,
            stderr=log_file,
            shell=True
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
                f"default_config.yaml"
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
    make_final_cmd,
    csv_row,
    row_number,
    input_csv_path,
    conda_env,
    run_locally,
    log_file_path,
    spreadsheet_url,
    worksheet_name, # This was none for some reason -- why?
    logger,
    lock,
    shared_rows_to_run,
    shared_default_config_paths,
    shared_csv_updates,
    current_step,
    total_rows
):

    final_cmd = None

    whether_to_run = csv_row[WHETHER_TO_RUN_COLUMN]

    if (
        whether_to_run.isnumeric()
            and int(whether_to_run) != 0
    ):

        replace_placeholders(csv_row, CURRENT_ROW_PLACEHOLDER, str(row_number))
        replace_placeholders(csv_row, CURRENT_WORKSHEET_PLACEHOLDER, worksheet_name)

        default_config_path_or_url = csv_row[PATH_TO_DEFAULT_CONFIG_COLUMN]
        if not default_config_path_or_url in shared_default_config_paths:
            with lock:
                default_config_path = fetch_default_config_path(
                    default_config_path_or_url,
                    logger
                )
                shared_default_config_paths[default_config_path_or_url] \
                    = default_config_path
        else:
            default_config_path \
                = shared_default_config_paths[default_config_path_or_url]

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

        log_folder = os.path.dirname(log_file_path)
        if not os.path.exists(log_folder):
            with lock:
                os.makedirs(log_folder, exist_ok=True)

        if run_locally:
            final_cmd = "{} &> {} &".format(cmd_as_string, log_file_path)
        else:
            final_cmd = make_final_cmd(
                csv_row,
                exp_name,
                log_file_path,
                cmd_as_string
            )

    if final_cmd is not None:

        shared_csv_updates.append((row_number, STATUS_CSV_COLUMN, SUBMITTED_STATUS))
        shared_csv_updates.append((row_number, WHETHER_TO_RUN_COLUMN, "0"))
        shared_rows_to_run.append(final_cmd)

    with lock:
        current_step.value += 1
        logger.progress(
            "Rows processing.",
            current_step.value,
            total_rows
        )


def make_final_cmd_slurm(csv_row, exp_name, log_file_path, cmd_as_string):

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
        with (NamedTemporaryFile('w', delete=False)) as tmp_file:
            fill_sbatch_script(tmp_file, slurm_args_dict, cmd_as_string)
            final_cmd = "sbatch {}".format(tmp_file.name)

    return final_cmd


def check_csv_column_names(csv_row, allowed_prefixes):

    assert MAIN_PATH_COLUMN in csv_row

    assert WHETHER_TO_RUN_COLUMN in csv_row

    assert PATH_TO_DEFAULT_CONFIG_COLUMN in csv_row

    for i, key in enumerate(csv_row.keys()):
        assert key is not None, \
            f"Column {i} has empty column name. " \
            f"Or some table entries contain commas."
        if PREFIX_SEPARATOR in key:
            assert any([prefix in key for prefix in allowed_prefixes]), \
                f"\"{key}\" does not contain any of allowed prefixes " \
                f"from:\n{allowed_prefixes}\n"


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
        DELTA_PREFIX + PREFIX_SEPARATOR,
        ignore_values=PLACEHOLDERS_FOR_DEFAULT
    )

    if len(deltas) > 0:
        check_duplicates(list(deltas.keys()))

    for key in deltas.keys():
        value = deltas[key]
        if value == EMPTY_STRING:
            deltas[key] = ""
        elif value == "":
            raise Exception(f"Empty value for {make_delta_column_name(key)}")

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
        if new_value is not None:
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

    all_slurm_args_dict = copy.deepcopy(DEFAULT_SLURM_ARGS_DICT)

    all_slurm_args_dict["job-name"] = exp_name

    all_slurm_args_dict["output"] = log_file
    all_slurm_args_dict["error"] = log_file

    specified_slurm_args = extract_from_csv_row_by_prefix(
        csv_row,
        SLURM_PREFIX + PREFIX_SEPARATOR,
        ignore_values=PLACEHOLDERS_FOR_DEFAULT
    )

    all_slurm_args_dict |= specified_slurm_args

    os.makedirs(os.path.dirname(all_slurm_args_dict["output"]), exist_ok=True)
    os.makedirs(os.path.dirname(all_slurm_args_dict["error"]), exist_ok=True)

    return all_slurm_args_dict


def extract_from_csv_row_by_prefix(csv_row, prefix, ignore_values):

    prefix_len = len(prefix)
    result = {}
    for key, value in csv_row.items():
        assert key is not None, "Possibly inconsistent number of delimeters."
        if key == prefix:
            raise Exception(
                f"Found \"{prefix}\" (nothing after this prefix) "
                f"in csv_row:\n{csv_row}"
            )
        if (
                len(key) > prefix_len
            and
                prefix == key[:prefix_len]
            and
                not value in ignore_values
        ):
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
