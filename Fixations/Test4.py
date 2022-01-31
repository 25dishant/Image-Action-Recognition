import imp
from torchvision.models import resnet50
import torch

resnet = resnet50(pretrained=False)
layers = list(resnet.children())[:7]

print(layers)
# model1 = torch.nn.Sequential(*(list(self.resnet.children())[:7]))