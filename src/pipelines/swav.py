import numpy as np
import torch
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.transforms.swav_transform import SwaVTransform
from lightly.data import LightlyDataset
from lightly import loss

def train_swav(model, config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    transform = SwaVTransform()

    train_data = LightlyDataset(input_dir=config.data_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            # num_workers=8,
        )

    val_data = LightlyDataset(input_dir=config.val_data_path, transform=transform)
    val_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            # num_workers=8,
        )

    model.to(device)
    criterion = SwaVLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_loss = []
    val_loss = []

    print("Starting Training")
    for epoch in range(config.epochs):
        
        epoch_loss = 0.0
        model.train()
        for batch in train_loader:
            views = batch[0]
            model.prototypes.normalize()
            multi_crop_features = [model(view.to(device)) for view in views]
            high_resolution = multi_crop_features[:2]
            low_resolution = multi_crop_features[2:]
            loss = criterion(high_resolution, low_resolution)
            epoch_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_val_loss = 0.0
        model.eval()
        for batch in val_loader:
            with torch.no_grad():
                views = batch[0]
                model.prototypes.normalize()
                multi_crop_features = [model(view.to(device)) for view in views]
                high_resolution = multi_crop_features[:2]
                low_resolution = multi_crop_features[2:]
                loss = criterion(high_resolution, low_resolution)
                epoch_val_loss += loss.detach()



        train_loss.append(epoch_loss.item())
        val_loss.append(epoch_val_loss.item())

        print(f"Epoch {epoch+1:02} - Train Loss: {epoch_loss.item()}, Validation Loss: {epoch_val_loss.item()}")

    return optimizer, {"train_loss":train_loss,"val_loss":val_loss}