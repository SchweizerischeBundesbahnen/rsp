import psutil


# source: https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
def human_readable_size(size, significant_places=3):
    """
    Print bytes in a human readable way with SI-style prefixes up to a number of significant digits.
    Parameters
    ----------
    size
    significant_places

    Returns
    -------

    """
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:  # noqa: B007
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{significant_places}f}{unit}"


def virtual_memory_human_readable() -> str:
    """Print virtual memory stats in a human readable format."""
    svmem = psutil.virtual_memory()
    return (
        f"total={human_readable_size(svmem.total)},"
        f"available={human_readable_size(svmem.available)},"
        f"percent={svmem.percent}%,"
        f"used={human_readable_size(svmem.used)},"
        f"free={human_readable_size(svmem.free)}"
    )


def current_process_stats_human_readable():
    """Print stats of current process in a human readable format."""
    p = psutil.Process()
    with p.oneshot():
        print(
            f"name={p.name()},"
            f"ppid={p.ppid()},"
            f"cpu_times={p.cpu_times()},"
            f"cpu_percent={p.cpu_percent()},"
            f"status={p.status()},"
            f"memory_info={p.memory_info()},"
        )
