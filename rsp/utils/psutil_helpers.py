import psutil


# source: https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
def human_readable_size(size, decimal_places=3):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:  # noqa: B007
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"


def virtual_memory_human_readable():
    svmem = psutil.virtual_memory()
    # print(svmem)
    print(f"total={human_readable_size(svmem.total)},"
          f"available={human_readable_size(svmem.available)},"
          f"percent={svmem.percent}%,"
          f"used={human_readable_size(svmem.used)},"
          f"free={human_readable_size(svmem.free)}")


def current_process_stats_human_readable():
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
