import torch
from utils.utils import NanException

if __name__ == "__main__":
    a = torch.Tensor(3,)
    raise NanException
    print(a)
