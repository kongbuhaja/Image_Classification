import random
import numpy as np
import torch
import os, psutil

def set_all(args):
    set_seed(args.seed)
    set_cpus(args.cpus)
    set_gpus(args.gpus)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)

def set_cpus(cpus):
    cores = [int(core) for core in cpus.split('-')]
    p = psutil.Process()
    p.cpu_affinity(list(range(cores[0], cores[1]+1)))

def set_gpus(gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus