from models.model_selector import get_model
from dataset import ImageDataLoader
from torch.utils.data import DataLoader
import numpy as np
import torch
import pickle 
from options import BaseOptions, Config, load_parameters
from utils import slice_image_paths

def load_model_weights(model, model_path,device):
    checkpoint = torch.load(model_path,map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    return model

opt = BaseOptions().parse()
config = Config(**load_parameters(opt.config_file))

# Load data
data = ImageDataLoader(config.dataset_path)
dataloader = DataLoader(data.dataset, batch_size=config.batch_size, shuffle=False)

# select embedding model training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = get_model(config.model)
model = load_model_weights(model, config.model_state_dict_path, device)
model.eval()


# compute embeddings and save
target = []
paths = []
labels = []
feature_embeddings = np.empty((0, config.embedding_dim))

for i, (x, y, path, label) in enumerate(dataloader):
    x = x.to(device=device)
    with torch.no_grad():
        batch_features = model(x)

    batch_features = batch_features.view(batch_features.size(0), -1).cpu().numpy()
    feature_embeddings = np.vstack((feature_embeddings, batch_features))
    target.extend(list(y.cpu().detach().numpy()))
    paths.extend(slice_image_paths(path))
    labels.extend(label)


data_dict = {
    "model": config.model,
    "embedding":feature_embeddings,
    "target":target,
    "paths": paths,
    "classes":labels
}

with open(config.save_file_path, 'wb') as pickle_file:
    pickle.dump(data_dict, pickle_file)