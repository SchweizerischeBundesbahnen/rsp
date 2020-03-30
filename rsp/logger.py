import logging
import sys

VERBOSE = 15

rsp_logger = logging.getLogger()
rsp_logger.setLevel(logging.INFO)
rsp_log_formatter = logging.Formatter(
    '[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s')

# standard handler to stdout
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(rsp_log_formatter)
rsp_logger.addHandler(stdout_handler)


# file handler
def add_file_handler_to_rsp_logger(file_name: str, log_level=logging.INFO):
    fh = logging.FileHandler(file_name)
    fh.setLevel(log_level)
    fh.setFormatter(rsp_log_formatter)
    rsp_logger.addHandler(fh)
