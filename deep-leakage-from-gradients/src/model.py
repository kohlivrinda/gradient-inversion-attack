import torch.nn as nn
import torchvision.models as models

class ModifiedResnet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet18()
        self._replace_relu_with_sigmoid()
        self._remove_strides()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def _replace_relu_with_sigmoid(self):
        replacements = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                replacements.append(name)

        for name in replacements:
            parent_module = self._get_parent_module(name)
            setattr(parent_module, name.split('.')[-1], nn.Sigmoid())

    def _remove_strides(self):
        for _, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.stride = (1, 1)
            elif isinstance(module, nn.MaxPool2d):
                module.stride = 1

    def _get_parent_module(self, name):
        """Helper function to get the parent module of a given module by name."""
        components = name.split('.')
        parent = self.model
        for comp in components[:-1]:  # Traverse down to the parent
            parent = getattr(parent, comp)
        return parent

    def forward(self, x):
        return self.model(x)



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 10)
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out