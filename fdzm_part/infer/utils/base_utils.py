import logging
import torch
import numpy as np
import random
import os
import math
import time


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_logger(logger_name=__name__, filename=None, filemode='a', add_console=False, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    fix_format = '%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] - %(name)s - %(message)s'
    logging.basicConfig(format=fix_format,
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=level,
                        filename=filename,
                        filemode=filemode)
    if add_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter(fix_format)
        console.setFormatter(formatter)
        # Create an instance
        logger.addHandler(console)
    return logger


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))



def get_evaluation_steps(num_train_steps, n_evaluations):
    """
    得到在那几步需要评估下效果
    """
    eval_steps = num_train_steps // n_evaluations
    eval_steps = [eval_steps * i for i in range(1, n_evaluations + 1)]
    eval_steps[-1] = num_train_steps
    return eval_steps
