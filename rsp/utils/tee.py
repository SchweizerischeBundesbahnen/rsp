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


def reset_tee(stdout_orig: multifile,
              stderr_orig: multifile,
              stdout_log_file: io.TextIOWrapper,
              stderr_log_file: io.TextIOWrapper):
    """Reset the stdout/stderr to the original.

    Parameters
    ----------
    stdout_orig
    stderr_orig
    """
    sys.stdout = stdout_orig
    sys.stderr = stderr_orig
    stdout_log_file.close()
    stderr_log_file.close()


def tee_stdout_stderr_to_file(stdout_log_file: str, stderr_log_file: str):
    """Redirect stdout and stderr to files as well.

    Parameters
    ----------
    stdout_log_file: str
        tee stdout to this file
    stderr_log_file
        tee stderr to this file
    Returns
    -------
    Tuple to pass to `reset_tee`
    """
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    stdout_log_file = open(stdout_log_file, 'a')
    sys.stdout = multifile([sys.stdout, stdout_log_file])
    stderr_log_file = open(stderr_log_file, 'a')
    sys.stderr = multifile([sys.stderr, stderr_log_file])
    return stdout_orig, stderr_orig, stdout_log_file, stderr_log_file
