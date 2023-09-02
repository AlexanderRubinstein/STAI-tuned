import time
from collections import defaultdict

from stuned.utility.logger import log_csv_for_concurrent
from stuned.utility.utils import retrier_factory


class GSheetBatchUpdater:
    def __init__(self, spreadsheet_url, worksheet_name, gsheet_client, logger, csv_path, update_interval=10):
        self.spreadsheet_url = spreadsheet_url
        self.worksheet_name = worksheet_name
        self.gsheet_client = gsheet_client
        self.logger = logger

        # Stores the updates in a queue as a nested dictionary: {row_id: {col: value}}
        self.queue = defaultdict(dict)
        self.update_interval = update_interval
        self.last_update = -1
        self.csv_path = csv_path
        # keep an internal flag to check if the last update was successful. If it was not, we can try again and skipping
        # the queue.
        self.last_update_status = False

    def add_to_queue(self, row, col, value):
        """Add an update to the queue."""
        self.queue[row][col] = value  # This will set the value for the specific column in the specific row

    def batch_update(self, force=False):
        """Batch update the Google Sheet with the queued changes."""
        if len(self.queue.keys()) == 0 or (not force and (self.last_update_status and time.time() - self.last_update < self.update_interval)):
            return True
        # Convert the queue to the report_stuff_ids format
        report_stuff_ids = [(row, col, value) for row, col_value_dict in self.queue.items() for col, value in
                            col_value_dict.items()]

        # Use retrier_factory to log the updates to the CSV
        try:
            retrier_factory(self.logger)(log_csv_for_concurrent)(self.csv_path, report_stuff_ids)
        except Exception as e:
            self.logger.log(f"Exception while logging to CSV: {e}")
            # don't clear the queue if logging to CSV failed
            # we can retry later
            self.logger.log("Not clearing the queue, will repeat later?")
            self.last_update_status = False
            return False

        affected_rows = list(self.queue.keys())

        self.queue.clear()  # Clear the queue after updating

        # Update the Google Sheet using your gsheet_client
        try:
            # self.gsheet_client.upload_csvs_to_spreadsheet(self.spreadsheet_url, [self.csv_path], [self.worksheet_name],
            #                                               single_rows_per_csv=[0])
            self.gsheet_client.upload_csvs_to_spreadsheet(self.spreadsheet_url, [self.csv_path], [self.worksheet_name],
                                                          single_rows_per_csv=[[0] + affected_rows])
        except Exception as e:
            self.logger.log(f"Exception while uploading to GSheet: {e}")
            self.logger.log("Will repeat later?")
            self.last_update_status = False
            return False

        self.last_update = time.time()  # Update the last update time
        self.last_update_status = True
        return True
