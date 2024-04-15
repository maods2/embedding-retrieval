from transformers import ViTForImageClassification
from torch import nn
import torch

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ModelAdapter:
    def __init__(self):
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.model.classifier = Identity()

    def __call__(self, x):
        self.model.eval() 
        output = self.model(x)
        return output.logits

    def to(self, device):
        self.model.to(device)

def get_vit():
    model = ModelAdapter()
    return model


