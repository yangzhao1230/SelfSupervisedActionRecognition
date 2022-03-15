import numpy as np
import logging
from config import *

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

class Log:
    @ex.capture
    def __init__(self, log_path) -> None:
        self.batch_data = dict()
        self.epoch_data = dict()
        self.logger = get_logger(log_path)
        self.logger.info('Start')

    def update_batch(self, name, value):
        if name not in self.batch_data:
            self.batch_data[name] =  list()
        self.batch_data[name].append(value)

    @ex.capture
    def update_epoch(self, epoch, epoch_num):
        self.logger.info('Epoch:[{}/{}]'.format(epoch + 1 , epoch_num))
        for name in self.batch_data.keys():
            if name not in self.epoch_data:
                self.epoch_data[name] = list()
            self.epoch_data[name].append(np.mean(self.batch_data[name]))
            self.batch_data[name] = list()
            self.logger.info("{}: {}".format(name, self.epoch_data[name][-1]))
