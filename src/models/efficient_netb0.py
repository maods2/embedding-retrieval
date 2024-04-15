from efficientnet_pytorch import EfficientNet
from torch import nn
import torch

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class CustomEncoder(nn.Module):
    def __init__(self, model,embedding_size=1280):
        super(CustomEncoder, self).__init__()
        self.encoder = model
        self.encoder._avg_pooling = nn.AdaptiveAvgPool2d(1)  # Modify average pooling
        self.encoder._fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embedding_size, 256 * 7 * 7)  # Adjust based on the specific EfficientNet variant
        )  # Modify the output layer

    def forward(self, x):
        return self.encoder(x).view(x.shape[0], 256, 7, 7)
    
def get_efficient_netb0():
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = Identity()


    return model

def get_efficient_netb0_encoder():
    model = CustomEncoder(
        model=EfficientNet.from_pretrained('efficientnet-b0'), 
        embedding_size=1280
        )
    return model



# Create an instance of the custom encoder
