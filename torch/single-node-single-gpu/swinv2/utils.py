import sys
import logging
import functools


@functools.lru_cache()
def create_logger(dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)

    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] ' + \
        '(%(filename)s %(lineno)d): ' + \
        '%(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:

        console_handler = logging.StreamHandler(sys.stdout)

        console_handler.setLevel(logging.DEBUG)

        console_handler.setFormatter(
            logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))

        logger.addHandler(console_handler)

    return logger
