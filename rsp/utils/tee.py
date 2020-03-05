# https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file
import io
import sys


class multifile(object):
    """Allows teeing."""

    def __init__(self, files):
        self._files = files

    def __getattr__(self, attr, *args):
        return self._wrap(attr, *args)

    def _wrap(self, attr, *args):
        def g(*a, **kw):
            for f in self._files:
                res = getattr(f, attr, *args)(*a, **kw)
            return res

        return g


def reset_tee(stdout_orig: io.TextIOWrapper):
    """Reset the stdout to the original.

    Parameters
    ----------
    stdout_orig
    """
    sys.stdout = stdout_orig


def tee_stdout_to_file(log_file: str):
    """

    Parameters
    ----------
    log_file: str
        tee to this file

    Returns
    -------
    Original stdout

    """
    print(f"log_file={log_file}")
    stdout_orig = sys.stdout
    sys.stdout = multifile([sys.stdout, open(log_file, 'w')])
    return stdout_orig
