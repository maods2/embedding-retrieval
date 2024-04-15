import torch
import torchvision.models as models



def get_vgg16():
    vgg16 = models.vgg16()
    vgg16.classifier = vgg16.classifier[:2]
    return vgg16


