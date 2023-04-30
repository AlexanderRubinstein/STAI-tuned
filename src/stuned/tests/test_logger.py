import sys
import os
import tensorboard as tb
import subprocess
import wandb
import datetime
import time
import filecmp


# local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utility.helpers_for_tests import (
    wrapper_for_test,
    assert_csv_diff,
    make_dummy_object
)
from utility.logger import (
    TB_URL_COLUMN,
    WANDB_URL_COLUMN,
    STATUS_CSV_COLUMN,
    COMPLETED_STATUS,
    WALLTIME_COLUMN,
    PATH_KEY,
    GDRIVE_FOLDER_KEY,
    delete_run_folders,
    redneck_logger_context,
    make_logger,
    try_to_log_in_tb,
    try_to_log_in_wandb,
    extract_csv_name_from_path,
    try_to_sync_csv_with_remote,
    sync_local_file_with_gdrive,
    extract_id_from_gdrive_url,
    make_gdrive_client,
    make_gspread_client,
    infer_logger_from_args
)
from utility.utils import (
    get_project_root_path,
    read_csv_as_dict,
    normalize_path,
    append_dict,
    assert_two_values_are_close,
    retrier_factory,
    get_current_time,
    get_value_from_config,
    remove_file_or_folder,
    touch_file,
    write_into_csv_with_column_names,
    make_file_lock
)


DELTA_SECONDS = 2
TEST_LOG_DICT = {
    'a': 0,
    'b': 1
}
TEST_RESULT_LOG_DICT = {
    'a': [0.0, 0.0],
    'b': [1.0, 1.0]
}
URL_SEP = '/'
SPACE_TOKEN = "%20"
TIME_TO_LOG_IN_WANDB = 5
TEST_DATA_FOLDER = os.path.join(
    get_project_root_path(),
    "src",
    "stuned",
    "tests",
    "data"
)
SYNC_TIME = 0.1
WAIT_FOR_DAEMON_TIME = 5


@wrapper_for_test
def test_logger_context_manager():

    def test_logger_context_manager_with_args(
        logger,
        start_time,
        logging_config,
        log_folder
    ):
        local_start_time = get_current_time()
        os.makedirs(log_folder, exist_ok=True)

        with redneck_logger_context(
            logging_config,
            log_folder,
            logger=logger,
            exp_name="Test experiment",
            start_time=start_time
        ) as logger:

            if logging_config["use_tb"]:
                assert logger.tb_run

                try_to_log_in_tb(logger, TEST_LOG_DICT, step=0)
                try_to_log_in_tb(logger, TEST_LOG_DICT, step=1)
            if logging_config["use_wandb"]:
                assert logger.wandb_run

                try_to_log_in_wandb(logger, TEST_LOG_DICT, step=0)
                try_to_log_in_wandb(logger, TEST_LOG_DICT, step=1)


        # do the checks
        assert logger.wandb_run is None
        assert logger.tb_run is None

        local_time_delta = get_current_time() - local_start_time

        tmp_inputs_path \
            = logging_config["output_csv"]["path"]

        csv_as_dict = read_csv_as_dict(tmp_inputs_path)
        csv_row_as_dict \
            = csv_as_dict[logging_config["output_csv"]["row_number"]]

        # check logged value
        if logging_config["use_tb"]:
            tb_exp_id = extract_tb_exp_id(csv_row_as_dict[TB_URL_COLUMN])

            tb_log_dict = logs_dict_from_tb_dataframe(
                read_tb_exp_logs(tb_exp_id)
            )
            assert_two_values_are_close(tb_log_dict, TEST_RESULT_LOG_DICT)

        if logging_config["use_wandb"]:
            if not logging_config["use_tb"]:
                time.sleep(TIME_TO_LOG_IN_WANDB)
            wandb_api_token = get_value_from_config(
                logging_config["wandb"]["netrc_path"],
                "password"
            )
            wandb_run = get_wandb_run_from_url(
                csv_row_as_dict[WANDB_URL_COLUMN],
                wandb_api_token
            )
            # assert wandb_run
            wandb_log_dict = logs_dict_from_wandb_dataframe(wandb_run.history())
            assert_two_values_are_close(wandb_log_dict, TEST_RESULT_LOG_DICT)

        # check completed status
        assert_two_values_are_close(
            csv_row_as_dict[STATUS_CSV_COLUMN],
            COMPLETED_STATUS
        )

        # check walltime
        walltime = datetime.datetime.strptime(
            csv_row_as_dict[WALLTIME_COLUMN],
            "%H:%M:%S.%f"
        ) - datetime.datetime.strptime("1900-01-01", "%Y-%m-%d")
        if start_time is None:
            assert walltime > datetime.timedelta(seconds=0)
            assert walltime < (get_current_time() - local_start_time)
        else:
            assert walltime.seconds == local_time_delta.seconds + DELTA_SECONDS

        # remove tb experiment on remote
        if logging_config["use_tb"] and logging_config["tb"]["upload_online"]:
            remove_tb_exp_from_remote(tb_exp_id)

        # remove wandb experiment on remote
        if logging_config["use_wandb"]:
            wandb_run.delete()
        # remove log folder
        delete_run_folders(tmp_inputs_path)
        # remove csv
        os.remove(tmp_inputs_path)

    tests_folder = os.path.dirname(TEST_DATA_FOLDER)
    log_folder = os.path.join(tests_folder, "tmp_log_folder")
    inputs_folder = os.path.join(get_project_root_path(), "inputs")
    tmp_csv_path = os.path.join(inputs_folder, "tmp_csv_for_logger.csv")
    logging_config = {
        "output_csv": {
            "path": tmp_csv_path,
            "row_number": 1,
            "spreadsheet_url": None,
            "worksheet_name": None
        },
        "use_tb": True,
        "tb": {
            "upload_online": True,
            "credentials_path": normalize_path(
                "~/.config/tensorboard/credentials/uploader-creds.json"
            )
        },
        "use_wandb": True,
        "wandb": {
            "netrc_path": normalize_path("~/.netrc")
        },
        GDRIVE_FOLDER_KEY: None
    }

    # both remote loggers, create logger inside
    test_logger_context_manager_with_args(
        logger=None,
        start_time=None,
        logging_config=logging_config,
        log_folder=log_folder
    )

    # create logger outside
    test_logger_context_manager_with_args(
        logger=make_logger(),
        start_time=None,
        logging_config=logging_config,
        log_folder=log_folder
    )

    # only tb as remote logger
    logging_config["use_wandb"] = False
    test_logger_context_manager_with_args(
        logger=None,
        start_time=None,
        logging_config=logging_config,
        log_folder=log_folder
    )

    # no remote loggers + non-zero start time
    logging_config["use_tb"] = False
    test_logger_context_manager_with_args(
        logger=None,
        start_time
            =(get_current_time() - datetime.timedelta(seconds=DELTA_SECONDS)),
        logging_config=logging_config,
        log_folder=log_folder
    )

    # only wandb as remote logger
    logging_config["use_wandb"] = True
    test_logger_context_manager_with_args(
        logger=None,
        start_time=None,
        logging_config=logging_config,
        log_folder=log_folder
    )


def read_tb_exp_logs(experiment_id):
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    return experiment.get_scalars()


def logs_dict_from_tb_dataframe(logs_df):
    result = {}
    df_dict = logs_df.to_dict()
    tags = list(df_dict["tag"].values())
    vals = list(df_dict["value"].values())
    for tag, val in zip(tags, vals):
        append_dict(result, {tag: val}, allow_new_keys=True)
    return result


def logs_dict_from_wandb_dataframe(logs_df):
    df_dict = logs_df.to_dict()
    df_dict.pop("_step")
    df_dict.pop("_runtime")
    df_dict.pop("_timestamp")
    return {key: list(value.values()) for key, value in df_dict.items()}


def extract_tb_exp_id(url):
    url_split = url.split(URL_SEP)
    assert len(url_split) > 1
    return url_split[-2]


def get_wandb_run_from_url(url, wandb_api_token):

    def extract_wandb_run_name(url):
        url_split = url.split(URL_SEP)
        assert len(url_split) > 3
        return f"{url_split[-4]}/{url_split[-3]}/{url_split[-1]}"

    api = wandb.Api(api_key=wandb_api_token)

    wandb_run_name = extract_wandb_run_name(url).replace(SPACE_TOKEN, " ")
    run = api.run(wandb_run_name)
    return run


@retrier_factory(make_logger())
def remove_tb_exp_from_remote(experiment_id):

    cmd_as_list = [
        "tensorboard",
        "dev",
        "delete",
        "--experiment_id",
        experiment_id
    ]
    try:
        output = subprocess.check_output(
            cmd_as_list,
            stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as exc:
        stderr_output = exc.output.decode("utf-8")
        raise Exception(
            "Subprocess for deleting tensorboard exp failed:"
            f"\n{stderr_output}"
        )
    assert "Deleted experiment" in output.decode("utf-8")


@wrapper_for_test
def test_gspread_sync():
    created_spreadsheet_folder = os.path.join(
        TEST_DATA_FOLDER,
        "created_spreadsheet"
    )
    os.makedirs(created_spreadsheet_folder, exist_ok=True)

    # create 2 csvs
    created_csv_one = os.path.join(
        created_spreadsheet_folder,
        "worksheet_1.csv"
    )
    created_csv_two = os.path.join(
        created_spreadsheet_folder,
        "worksheet_2.csv"
    )
    touch_file(created_csv_one)
    touch_file(created_csv_two)

    write_into_csv_with_column_names(
        created_csv_one,
        1,
        "A",
        "1"
    )
    write_into_csv_with_column_names(
        created_csv_one,
        1,
        "B",
        "2"
    )
    write_into_csv_with_column_names(
        created_csv_two,
        1,
        "C",
        "3"
    )
    write_into_csv_with_column_names(
        created_csv_two,
        1,
        "D",
        "4"
    )

    logger = make_logger()

    gspread_client = make_gspread_client(logger)

    created_csvs = [created_csv_one, created_csv_two]

    spreadsheet_url = gspread_client.upload_csvs_to_spreadsheet(
        None,
        created_csvs,
        worksheet_names=None
    )

    # rename column in first
    write_into_csv_with_column_names(
        created_csv_one,
        0,
        "B",
        "B_new"
    )

    # update value in second
    write_into_csv_with_column_names(
        created_csv_two,
        1,
        "D",
        "100500"
    )

    # sync
    output_csv_config = {
        PATH_KEY: created_csv_one,
        "row_number": None,
        "spreadsheet_url": spreadsheet_url,
        "worksheet_name": extract_csv_name_from_path(
            created_csv_one
        )
    }
    logger.set_csv_output(
        output_csv_config
    )
    try_to_sync_csv_with_remote(logger)
    output_csv_config[PATH_KEY] = created_csv_two
    output_csv_config["worksheet_name"] = extract_csv_name_from_path(
        created_csv_two
    )
    logger.set_csv_output(
        output_csv_config
    )
    try_to_sync_csv_with_remote(logger)

    downloaded_spreadsheet_folder = os.path.join(
        TEST_DATA_FOLDER,
        "downloaded_spreadsheet"
    )

    gspread_client.download_spreadsheet_as_csv(
        spreadsheet_url=spreadsheet_url,
        folder_for_csv=downloaded_spreadsheet_folder
    )

    # assert the same csvs
    for created_csv, downloaded_csv in zip(
        created_csvs,
        sorted(os.listdir(downloaded_spreadsheet_folder))
    ):
        downloaded_csv = os.path.join(
            downloaded_spreadsheet_folder,
            downloaded_csv
        )
        assert os.path.basename(created_csv) == os.path.basename(downloaded_csv)
        assert_csv_diff(
            created_csv,
            downloaded_csv
        )

    # remove all created files
    remove_file_or_folder(created_spreadsheet_folder)
    remove_file_or_folder(downloaded_spreadsheet_folder)
    gspread_client.delete_spreadsheet(spreadsheet_url)


@wrapper_for_test
def test_gdrive_sync():

    logger = make_logger()

    @retrier_factory(logger)
    def sync_and_compare(
        gdrive_client,
        local_filepath,
        downloaded_filepath,
        remote_file_url,
        remote_folder_id
    ):

        remote_file = sync_local_file_with_gdrive(
            gdrive_client,
            downloaded_filepath,
            remote_file_url,
            download=True,
            logger=None
        )

        assert remote_file["parents"][0]["id"] == remote_folder_id

        assert filecmp.cmp(
            local_filepath,
            downloaded_filepath,
            shallow=False
        )

    local_folder = os.path.join(
        TEST_DATA_FOLDER,
        "dummy_log_folder"
    )
    os.makedirs(local_folder, exist_ok=True)
    dummy_stdout_path = os.path.join(local_folder, "stdout.txt")
    dummy_stderr_path = os.path.join(local_folder, "stderr.txt")

    gdrive_client = make_gdrive_client(logger)

    downloaded_dummy_stdout_path = os.path.join(
        TEST_DATA_FOLDER,
        "downloaded_stdout"
    )

    downloaded_dummy_stderr_path = os.path.join(
        TEST_DATA_FOLDER,
        "downloaded_stderr"
    )

    remote_test_storage = gdrive_client.create_node(
        "test_storage",
        node_type="folder"
    )

    logger.update_output_folder(local_folder)

    remote_logs_folder_url, remote_stdout_url, remote_stderr_url \
        = logger.create_logs_on_gdrive(
            remote_test_storage["embedLink"]
        )

    logger.start_gdrive_daemon(sync_time=SYNC_TIME)

    remote_folder_id = extract_id_from_gdrive_url(remote_logs_folder_url)
    remote_folder = gdrive_client.get_node_by_id(remote_folder_id)
    remote_folder["parents"][0]["id"] == remote_test_storage["id"]

    with make_file_lock(dummy_stdout_path):
        print("Dummy stdout output", file=open(dummy_stdout_path, 'a'))
    with make_file_lock(dummy_stderr_path):
        print("Dummy stderr output", file=open(dummy_stderr_path, 'a'))
    time.sleep(WAIT_FOR_DAEMON_TIME)

    sync_and_compare(
        gdrive_client,
        dummy_stdout_path,
        downloaded_dummy_stdout_path,
        remote_stdout_url,
        remote_folder_id
    )
    sync_and_compare(
        gdrive_client,
        dummy_stderr_path,
        downloaded_dummy_stderr_path,
        remote_stderr_url,
        remote_folder_id
    )

    with make_file_lock(dummy_stdout_path):
        print("New to stdout", file=open(dummy_stdout_path, 'a'))
    with make_file_lock(dummy_stderr_path):
        print("New to stderr", file=open(dummy_stderr_path, 'a'))
    time.sleep(WAIT_FOR_DAEMON_TIME)

    sync_and_compare(
        gdrive_client,
        dummy_stdout_path,
        downloaded_dummy_stdout_path,
        remote_stdout_url,
        remote_folder_id
    )
    sync_and_compare(
        gdrive_client,
        dummy_stderr_path,
        downloaded_dummy_stderr_path,
        remote_stderr_url,
        remote_folder_id
    )

    logger.stop()

    remote_test_storage.Delete()

    remove_file_or_folder(local_folder)
    remove_file_or_folder(downloaded_dummy_stdout_path)
    remove_file_or_folder(downloaded_dummy_stderr_path)


@wrapper_for_test
def test_logger_inference_from_args():

    def check(logger, inferred_logger):
        assert inferred_logger is logger
        return None

    logger = make_logger()

    # self as complex object
    complex_object = make_dummy_object()
    complex_object.logger = logger
    inferred_logger = infer_logger_from_args(complex_object, "a", "b", key="arg")
    inferred_logger = check(logger, inferred_logger)

    # self as logger
    inferred_logger = infer_logger_from_args(logger, "a", "b", key="arg")
    inferred_logger = check(logger, inferred_logger)

    # key arg
    inferred_logger = infer_logger_from_args("a", "b", logger=logger)
    inferred_logger = check(logger, inferred_logger)

    # positional arg
    inferred_logger = infer_logger_from_args("a", logger, key="b")
    inferred_logger = check(logger, inferred_logger)

    # key arg
    inferred_logger = infer_logger_from_args("a", "b", key=logger)
    assert inferred_logger is None


def main():
    # test_gspread_sync()
    # test_gdrive_sync()
    test_logger_context_manager()
    test_logger_inference_from_args()


if __name__ == "__main__":
    main()
