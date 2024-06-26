from datetime import datetime
import sys


from utils.dataset import ImageDataLoader
from torch.utils.data import DataLoader
from utils.utils import slice_image_paths
import matplotlib.pyplot as plt
import pickle
import torch
import torchvision
from efficientnet_pytorch import EfficientNet
import numpy as np

from lightly import loss

from lightly.data import LightlyDataset

from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.transforms.swav_transform import SwaVTransform



class SwaV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.backbone._fc = torch.nn.Identity()

        self.projection_head = SwaVProjectionHead(1280, 1280, 128)
        self.prototypes = SwaVPrototypes(128, n_prototypes=1280)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = torch.nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p


def get_swav():
    model = SwaV()
    return model