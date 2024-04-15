from transformers import CLIPModel, CLIPProcessor
from torch import nn
import torch

class ModelAdapter:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


    def __call__(self, x):
        self.model.eval() 
        output = self.model.get_image_features(x)
        output /= output.norm(dim=-1, keepdim=True)
        return output

    def to(self, device):
        self.model.to(device)

def get_clip_vit():
    model = ModelAdapter()
    return model


