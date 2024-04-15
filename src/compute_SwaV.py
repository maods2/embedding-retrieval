from datetime import datetime
import sys

# sys.path.append('../embeddings/')
# sys.path.append('./embeddings/')
sys.path.append('..')

from dataset import ImageDataLoader
from torch.utils.data import DataLoader
from utils import slice_image_paths
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

ABSOLUTE_PATH = 'C:/Users/Maods/Documents/Development/Mestrado/terumo/apps/embedding-retrieval/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class SwaV(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SwaVProjectionHead(1280, 1280, 128)
        self.prototypes = SwaVPrototypes(128, n_prototypes=1280)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = torch.nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p


# Use a resnet backbone.
backbone = EfficientNet.from_pretrained('efficientnet-b0')
# Ignore the classification head as we only want the features.
backbone._fc = torch.nn.Identity()


model = SwaV(backbone)



caminho_do_arquivo = f'{ABSOLUTE_PATH}checkpoints/embedding_models/efficientnet_SwaV_model_test2.pth'
model = torch.load(caminho_do_arquivo, map_location=device)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


model.to(device)
model.eval()

# compute embeddings and save
data_paths = [('train', f'{ABSOLUTE_PATH}data/terumo-data-training'), ('test', f'{ABSOLUTE_PATH}data/terumo-data-testset')]

for method, path in data_paths:

    data = ImageDataLoader(path)
    dataloader = DataLoader(data.dataset, batch_size=50, shuffle=False)

    target = []
    paths = []
    labels = []
    feature_embeddings = np.empty((0, 1280))



    for i, (x, y, path, label) in enumerate(dataloader):
        x = x.to(device=device)
        with torch.no_grad():
            batch_features = model(x)

        batch_features_np = batch_features.view(batch_features.size(0), -1).cpu().numpy()
        feature_embeddings = np.vstack((feature_embeddings, batch_features_np))
        target.extend(list(y.cpu().detach().numpy()))
        paths.extend(slice_image_paths(path))
        labels.extend(label)


    data_dict = {
        "model": 'efficientnet_SwaV',
        "embedding":feature_embeddings,
        "target":target,
        "paths": paths,
        "classes":labels
    }

    with open(f'{ABSOLUTE_PATH}embeddings/efficientnet_SwaV_{method}.pickle', 'wb') as pickle_file:
        pickle.dump(data_dict, pickle_file)


