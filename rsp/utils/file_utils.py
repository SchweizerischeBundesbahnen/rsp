"""File utils."""
import errno
import os


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
