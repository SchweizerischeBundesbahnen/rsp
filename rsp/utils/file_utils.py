"""File utils."""
import errno
import os
import re


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
    # the last int in filename corresponds to the experiment id
    return int(re.findall(r'\d+', filename)[-1])
