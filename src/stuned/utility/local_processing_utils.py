import os


def process_exists(pid):
    """Check if a process with the given PID exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True