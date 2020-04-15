import os
import sys
import math
import fire
import json
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing
import torch.backends.cudnn as cudnn

import numpy as np
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F

from torch_optimizer import DiffGrad
from torch.autograd import grad as torch_grad

import torchvision
from torchvision import transforms

from PIL import Image
from pathlib import Path
from utils.utils import *
from  model.Conv2DMod import *
from  model.Generator import *
from  model.Discriminator import *
from  model.StyleVectorizer import *

class StyleGAN2(nn.Module):
    def __init__(self, image_size, latent_dim = 512, style_depth = 8, network_capacity = 16, transparent = False, steps = 1, lr = 1e-4):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)

        self.S = StyleVectorizer(latent_dim, style_depth)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent = transparent)
        self.D = Discriminator(image_size, network_capacity, transparent = transparent)

        self.SE = StyleVectorizer(latent_dim, style_depth)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent = transparent)

        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = DiffGrad(generator_params, lr = self.lr, betas=(0.5, 0.9))
        self.D_opt = DiffGrad(self.D.parameters(), lr = self.lr, betas=(0.5, 0.9))

        self._init_weights()
        self.reset_parameter_averaging()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x
