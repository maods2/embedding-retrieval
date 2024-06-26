# Inspiration
# https://towardsdatascience.com/a-hands-on-introduction-to-image-retrieval-in-deep-learning-with-pytorch-651cd6dba61e

import os
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from torch import nn
import torch.optim as optim


class TripletData(Dataset):
    def __init__(self, path, transform, config):
        self.path = path
        self.categories_num = config.triplet_categories_num
        self.transform = transform
        self.image_extensions = ["jpg", "jpeg", "png", "gif", "bmp", "tiff"]
        self.class_mapping = {}
        for i, label in enumerate(config.triplet_dataset_classes):
            self.class_mapping[i+1] = label


    def __getitem__(self, idx):
        # our positive class for the triplet
        idx = idx % self.categories_num + 1

        # choosing our pair of positive images (im1, im2)
        positive_data_dir = Path(os.path.join(
            self.path, self.class_mapping[idx]))
        positives = [file for file in positive_data_dir.glob(
            '**/*') if file.suffix.lower()[1:] in self.image_extensions]
        im1, im2 = random.sample(positives, 2)

        # choosing a negative class and negative image (im3)
        negative_categories = list(self.class_mapping.values())
        negative_categories.remove(self.class_mapping[idx])
        negative_category = str(random.choice(negative_categories))
        negative_data_dir = Path(os.path.join(self.path, negative_category))
        negatives = [file for file in negative_data_dir.glob(
            '**/*') if file.suffix.lower()[1:] in self.image_extensions]
        im3 = random.choice(negatives)

        im1 = self.transform(Image.open(im1))
        im2 = self.transform(Image.open(im2))
        im3 = self.transform(Image.open(im3))

        return [im1, im2, im3]

    # we'll put some value that we want since there can be far too many triplets possible
    # multiples of the number of images/ number of categories is a good choice
    def __len__(self):
        return self.categories_num*50


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    # Distances in embedding space is calculated in euclidean
    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive -
                            distance_negative + self.margin)
        return losses.mean()


def train_triplet(model, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_data = TripletData(config.data_path, transform, config)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=config.batch_size, shuffle=True)

    val_data = TripletData(config.val_data_path, transform, config)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_data, batch_size=config.batch_size, shuffle=False)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    triplet_loss = TripletLoss()

    train_loss = []
    val_loss = []
    # Training
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for data in train_loader:

            optimizer.zero_grad()
            x1, x2, x3 = data
            e1 = model(x1.to(device))
            e2 = model(x2.to(device))
            e3 = model(x3.to(device))

            loss = triplet_loss(e1, e2, e3)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        epoch_val_loss = 0.0
        model.eval()
        for data in val_loader:
            with torch.no_grad():

                x1, x2, x3 = data
                e1 = model(x1.to(device))
                e2 = model(x2.to(device))
                e3 = model(x3.to(device))

                loss = triplet_loss(e1, e2, e3)
                epoch_val_loss += loss



        train_loss.append(epoch_loss.item())
        val_loss.append(epoch_val_loss.item())

        print(f"Epoch {epoch+1:02} - Train Loss: {epoch_loss.item()}, Validation Loss: {epoch_val_loss.item()}")

    return optimizer, {"train_loss":train_loss,"val_loss":val_loss}