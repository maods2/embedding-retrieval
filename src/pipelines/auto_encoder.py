# Inspiration
# https://medium.com/pytorch/image-similarity-search-in-pytorch-1a744cf3469

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from pathlib import Path
from PIL import Image
from torch import nn
from models.auto_encoders import ConvDecoder
import torch
import torchvision.transforms as T
import torch.optim as optim
import sys



class FolderDataset(Dataset):
    """
    Creates a PyTorch dataset from folder, returning two tensor images.
    Args: 
    main_dir : directory where images are stored.
    transform (optional) : torchvision transforms to be applied while making dataset
    """

    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = self._get_images_path(main_dir)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx: int):
        """Returns the dataset item with that given index, for the autoencoder training.

        Args:
            idx (int): index of image to retrieve

        Returns:
            (torch.tensor, torch.tensor): returns a tuple containing the training image for that
            given index, twice (i.e duplicated). This will be used to train the autoencoder, since
            its objective is for the decoder to learn to reverse the transformation done by the encoder.
        """
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")

        if self.transform is not None:
            tensor_image = self.transform(image)

        return tensor_image, tensor_image
    
    def _get_images_path(self, main_dir):
        image_extensions = ["jpg", "jpeg", "png", "gif", "bmp", "tiff"]
        data_dir = Path(main_dir)
        images_path = [file for file in data_dir.glob(
            '**/*') if file.suffix.lower()[1:] in image_extensions]
        return images_path
    

def train_step(encoder, decoder, train_loader, loss_fn, optimizer, device):
    """
    Performs a single training step
    Args:
    encoder: A convolutional Encoder. E.g. torch_model ConvEncoder
    decoder: A convolutional Decoder. E.g. torch_model ConvDecoder
    train_loader: PyTorch dataloader, containing (images, images).
    loss_fn: PyTorch loss_fn, computes loss between 2 images.
    optimizer: PyTorch optimizer.
    device: "cuda" or "cpu"
    Returns: Train Loss
    """
    #  Set networks to train mode.
    encoder.train()
    decoder.train()

    for batch_idx, (train_img, target_img) in enumerate(train_loader):
        # Move images to device
        train_img = train_img.to(device)
        target_img = target_img.to(device)
        
        # Zero grad the optimizer
        optimizer.zero_grad()
        # Feed the train images to encoder
        enc_output = encoder(train_img)
        # The output of encoder is input to decoder !
        dec_output = decoder(enc_output)
        
        # Decoder output is reconstructed image
        # Compute loss with it and orginal image which is target image.
        loss = loss_fn(dec_output, target_img)
        # Backpropogate
        loss.backward()
        # Apply the optimizer to network by calling step.
        optimizer.step()
    # Return the loss
    return loss.item()

def val_step(encoder, decoder, val_loader, loss_fn, device):
    """
    Performs a single training step
    Args:
    encoder: A convolutional Encoder. E.g. torch_model ConvEncoder
    decoder: A convolutional Decoder. E.g. torch_model ConvDecoder
    val_loader: PyTorch dataloader, containing (images, images).
    loss_fn: PyTorch loss_fn, computes loss between 2 images.
    device: "cuda" or "cpu"
    Returns: Validation Loss
    """
    
    # Set to eval mode.
    encoder.eval()
    decoder.eval()
    
    # We don't need to compute gradients while validating.
    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(val_loader):
            # Move to device
            train_img = train_img.to(device)
            target_img = target_img.to(device)

            # Again as train. Feed encoder the train image.
            enc_output = encoder(train_img)
            # Decoder takes encoder output and reconstructs the image.
            dec_output = decoder(enc_output)

            # Validation loss for encoder and decoder.
            loss = loss_fn(dec_output, target_img)
    # Return the loss
    return loss.item()



def train_auto_encoder(model, config):
   
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
    ])

    full_dataset = FolderDataset(config.data_path, transform) # Create folder dataset.

    train_size = 0.75
    val_size = 1 - train_size

    # Split data to train and test
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size]) 

    # Create the train dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Create the validation dataloader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size)

    loss_fn = nn.MSELoss() # We use Mean squared loss which computes difference between two images.

    encoder = model # Our encoder model
    decoder = ConvDecoder() # Our decoder model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # Shift models to GPU
    encoder.to(device)
    decoder.to(device)

    # Both the enocder and decoder parameters
    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(autoencoder_params, lr=config.lr) # Adam Optimizer

    t_loss = []
    v_loss = []
    # Usual Training Loop
    for epoch in range(config.epochs):
        train_loss = train_step(encoder, decoder, train_loader, loss_fn, optimizer, device=device)
        
        val_loss = val_step(encoder, decoder, val_loader, loss_fn, device=device)
        
        print(f"Epochs = {epoch}, Training Loss : {train_loss:.2f}, Validation Loss : {val_loss:.2f}")
        t_loss.append(train_loss)
        v_loss.append(val_loss)

    return optimizer, {"train_loss":t_loss,"val_loss":v_loss}