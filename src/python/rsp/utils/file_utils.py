"""File utils."""
import errno
import os
import re
import sys
import time


def newline_and_flush_stdout_and_stderr():
    sys.stderr.write("\n")
    sys.stderr.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    # give flushing a bit of time to finish...
    time.sleep(0.1)


def check_create_folder(folder_name):
    """Checks that the folder exists. Tries to create the folder if it does not
    exist.

    Parameters
    ----------
    folder_name
    """
    if not os.path.exists(folder_name):
        try:
            os.makedirs(folder_name)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise exc


def get_experiment_id_from_filename(filename: str) -> int:
    """Extracts the experiment id as an int from filename.

    Parameters
    ----------
    filename

    Returns
    -------
    """
    # the match the 4-digit group after experimemnt_0004_.....pkl
    return int(re.findall(r"experiment_([0-9]{4})", filename)[-1])
