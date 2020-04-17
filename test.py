import torch
from utils.utils import NanException

import numpy as np


if __name__ == "__main__":
    init_dim = 8
    n_tile = 8
    t_l = [init_dim * np.arange(n_tile) + i for i in range(init_dim)]
    print(t_l)
