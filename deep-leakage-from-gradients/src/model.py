import torch.nn
import torch.nn as nn
from torchvision import models, datasets, transforms


class ModifiedResnet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet18()

        for _, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                setattr(self.model, nn.Sigmoid())

        self._remove_strides()

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def _remove_strides(self):
        for _, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.stride = (1,1)
            elif isinstance(module, nn.MaxPool2d):
                module.stride = 1
    
    def forward(self, x):
        return self.model(x)
    
    



