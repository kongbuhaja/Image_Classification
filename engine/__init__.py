from .argparse import arg_parser
args = arg_parser()

from .env import set_all
set_all(args)

from .train import process as train_process
from .eval import process as eval_process

__all__ = ['arg_parser',
           'set_all',
           'args',
           'train_process',
           'eval_process']