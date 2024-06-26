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
import re
from PIL import Image
from torch import nn
from torchvision import transforms
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_model_weights(model, model_path,device):
    checkpoint = torch.load(model_path,map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    return model

def get_all_image_files(pathlib_root_folder):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    img_regex = re.compile('|'.join(img_extensions), re.IGNORECASE)
    image_files = [f for f in pathlib_root_folder.glob('**/*') if f.is_file() and img_regex.search(f.suffix)]
    return image_files






class Net(nn.Module):
    def __init__(self, net_version, num_classes, freeze: bool = False):
        super(Net, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-'+net_version)
        self.backbone._fc = nn.Sequential(
            nn.Linear(1280, num_classes),
        )
        if freeze:
            # freeze backbone layers
            for name, param in self.backbone.named_parameters():
                if not name.startswith("_fc"):
                    param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)
    

def compute_semantic_attributes(config, mode):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    models = []
    for filename in config.semantic_models_path:
        model_path = config.semantic_basepath + filename
        models.append(
            load_model_weights(Net(net_version="b0", num_classes=2).to(device), model_path, device)
        )
    
    data = ImageDataLoader(config.data_path)
    dataloader = DataLoader(data.dataset, batch_size=config.batch_size, shuffle=False)

    target = []
    paths = []
    labels = []
    num_att = config.num_att
    feature_embeddings = np.empty((0, num_att))




    for i, (x, y, path, label) in enumerate(dataloader):
        x = x.to(device=device)
        with torch.no_grad():
            prediction_columns = []
            for model in models:
                model.eval()
                output = model(x)
                scores = torch.sigmoid(output)
                prediction_columns.append(scores[:, 0].view(-1, 1))

        prediction_matrix = torch.cat(prediction_columns, dim=1)

          
        emb_batch = prediction_matrix.cpu().detach().numpy()
        feature_embeddings = np.vstack((feature_embeddings, emb_batch))
        target.extend(list(y.cpu().detach().numpy()))
        paths.extend(slice_image_paths(path))
        labels.extend(label)

        print(f"{i} of {len(dataloader)} batchs")

    data_dict = {
        "model": 'semantic_att',
        "embedding":feature_embeddings,
        "target":target,
        "paths": paths,
        "classes":labels
    }



    with open(f'{config.save_embedding_path}_{mode}.pickle', 'wb') as pickle_file:
        pickle.dump(data_dict, pickle_file)



