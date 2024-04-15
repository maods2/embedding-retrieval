from datetime import datetime
import sys

# sys.path.append('../embeddings/')
# sys.path.append('./embeddings/')
# sys.path.append('C:/Users/Maods/Downloads/terumo-seg-esclerose-main/terumo-seg-esclerose-main/terumo_seg_esclerose/')
# sys.path.append('C:/Users/Maods/Downloads/terumo-seg-esclerose-main/terumo-seg-esclerose-main/terumo_seg_esclerose/utils')
# sys.path.append('..')

from dataset import ImageDataLoader
from torch.utils.data import DataLoader
from utils import slice_image_paths
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

import typer
from terumo_seg_esclerose.utils.config import ConfigLoaded

from terumo_seg_esclerose.predict.seg_glomerulus import (
    predict as predict_glo,
    get_model_list as get_model_list_glo
    )
from terumo_seg_esclerose.predict.seg_sclerosis import( 
    predict as predict_sle,
    get_model_list as get_model_list_sle,
    )
from mmengine.config import Config
import numpy as np
app = typer.Typer()

ABSOLUTE_PATH = 'C:/Users/Maods/Documents/Development/Mestrado/terumo/apps/embedding-retrieval/'

path = "C:/Users/Maods/Documents/Development/Mestrado/terumo/apps/terumo-seg-esclerose-main/configs/tiny/tiny-efficientnetb0-unet-pipeline.py"
ConfigLoaded().load_config(path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'CUDA: {device}')
# from terumo_seg_esclerose.cli import run_predict


def load_model_weights(model, model_path,device):
    checkpoint = torch.load(model_path,map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    return model

def get_all_image_files(pathlib_root_folder):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    img_regex = re.compile('|'.join(img_extensions), re.IGNORECASE)
    image_files = [f for f in pathlib_root_folder.glob('**/*') if f.is_file() and img_regex.search(f.suffix)]
    return image_files


def predict(model, image):
    model.eval()

    with torch.no_grad():
        
        output = model(image.to(device))
        scores = torch.sigmoid(output)
        predictions = (scores>0.5).float()
        _, pred = torch.min(predictions, 1)

    return pred.item()

def sclerosis_predict(image_path, model_list_glo, model_list_sle):
    mask_glo = predict_glo(image_path, model_list_glo)
    mask_sle = predict_sle(image_path, model_list_sle)

    inter = np.logical_and(mask_glo > 0.5, mask_sle > 0.5)
    p = np.sum(inter) / (np.sum(mask_glo > 0.5) + 0.00001)

    # print(f"Glomerulu with {p} sclerosis")
    return p



model_list_glo = get_model_list_glo()
model_list_sle = get_model_list_sle()



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
    



hiper_path = f'{ABSOLUTE_PATH}checkpoints/semantic_models/Hypercellularity-2024-02-10 21_19_55.851047-3_fold_min_loss_checkpoint.pth.tar'
membran_path = f'{ABSOLUTE_PATH}checkpoints/semantic_models/Membranous-2024-02-10 21_20_09.125156-1_fold_min_loss_checkpoint.pth.tar'
sclero_path = f'{ABSOLUTE_PATH}checkpoints/semantic_models/Sclerosis-2024-02-10 21_20_15.760971-3_fold_min_loss_checkpoint.pth.tar'
normal_path = f'{ABSOLUTE_PATH}checkpoints/semantic_models/Normal-2024-02-10 21_20_02.347231-2_fold_min_loss_checkpoint.pth.tar'
podoc_path = f'{ABSOLUTE_PATH}checkpoints/semantic_models/Podocytopathy-2024-02-10 21_20_29.072759-2_fold_min_loss_checkpoint.pth.tar'
cresc_path = f'{ABSOLUTE_PATH}checkpoints/semantic_models/Crescent-2024-02-10 21_20_22.470548-1_fold_min_loss_checkpoint.pth.tar'


hiper_model = load_model_weights(Net(net_version="b0", num_classes=2).to(device), hiper_path,device)
membran_model = load_model_weights(Net(net_version="b0", num_classes=2).to(device), membran_path,device)
sclero_model = load_model_weights(Net(net_version="b0", num_classes=2).to(device), sclero_path,device)
normal_model = load_model_weights(Net(net_version="b0", num_classes=2).to(device), normal_path,device)
podoc_model = load_model_weights(Net(net_version="b0", num_classes=2).to(device), podoc_path,device)
cresc_model = load_model_weights(Net(net_version="b0", num_classes=2).to(device), cresc_path,device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


models = [hiper_model,membran_model,sclero_model,normal_model,podoc_model,cresc_model]



data = ImageDataLoader(f'{ABSOLUTE_PATH}data/terumo-data-training')
dataloader = DataLoader(data.dataset, batch_size=128, shuffle=False)

target = []
paths = []
labels = []
num_att = 7
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

    sclerosis_batch = np.array([sclerosis_predict(
                                            p,
                                            model_list_glo,
                                            model_list_sle
                                                  ) for p in path]).reshape(-1,1)     
    emb_batch = np.hstack(( prediction_matrix.cpu().detach().numpy(), sclerosis_batch))
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



with open(f'{ABSOLUTE_PATH}embeddings/semantic_train.pickle', 'wb') as pickle_file:
    pickle.dump(data_dict, pickle_file)



data = ImageDataLoader(f'{ABSOLUTE_PATH}data/terumo-data-testset')
dataloader = DataLoader(data.dataset, batch_size=50, shuffle=False)

target = []
paths = []
labels = []
num_att = 7
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
    sclerosis_batch = np.array([sclerosis_predict(
                                            p,
                                            model_list_glo,
                                            model_list_sle
                                                  ) for p in path]).reshape(-1,1)    
    emb_batch = np.hstack(( prediction_matrix.cpu().detach().numpy(), sclerosis_batch))
    feature_embeddings = np.vstack((feature_embeddings, emb_batch))

    target.extend(list(y.cpu().detach().numpy()))
    paths.extend(slice_image_paths(path))
    labels.extend(label)


data_dict = {
    "model": 'semantic_att',
    "embedding":feature_embeddings,
    "target":target,
    "paths": paths,
    "classes":labels
}



with open(f'{ABSOLUTE_PATH}embeddings/semantic_test.pickle', 'wb') as pickle_file:
    pickle.dump(data_dict, pickle_file)