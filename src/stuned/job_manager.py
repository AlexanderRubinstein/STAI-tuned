"""
TODO: make this functional
Job manager that manages jobs and keeps track of their status.
Ideally works for both local and remote runs.
"""
class JobManager():
    def __init__(self, local_run):
        self.jobs = {}
        self.local_run = local_run
    @property
    def get_running_jobs_count(self):
        pass

    @property
    def get_finished_jobs_count(self):
        pass

    @property
    def get_failed_jobs_count(self):
        pass

