import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from utils import label_to_onehot, cross_entropy_for_onehot
from model import ModifiedResnet18
import matplotlib.pyplot as plt

torch.manual_seed(24)

dataset = datasets.Cifar10(root='../data', download = True)
data_transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor()
    ])


device = "cuda" if torch.cuda.is_available() else 'cpu'
model = ModifiedResnet18().to(device)
criterion = cross_entropy_for_onehot
img_idx = 22

gt_img = data_transforms(dataset[img_idx][0].to(device)) #(C,H,W)
gt_img = gt_img.view(1, *gt_img.size()) #(N,C,H,W)
gt_label = torch.Tensor([dataset[img_idx][1]]).long().to(device) # scalar 
gt_label = gt_label.view(1, ) # (N, ) expected by loss function
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

#TODO: visualize plot for gt_img
plt.imshow(transforms.ToPILImage((gt_img[0].cpu())))

# original gradient calculation
pred = model(gt_img)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, model.parameters())

#share gradient with other clients
og_dy_dx = list((_.detach().clone() for _ in dy_dx))

# create dummy data
dummy_data = torch.randn(gt_img.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

plt.imshow(transforms.ToPILImage((dummy_data[0].cpu())))

#DLG Algo:

opt = torch.optim.LBFGS([dummy_data, dummy_label])
history = []

for iters in range(3000):
    def closure():
        opt.zero_grad()
        dummy_pred = model(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, og_dy_dx):
            grad_diff += ((gx-gy)**2).sum()
        grad_diff.backward()
        return grad_diff
    
    opt.step(closure)

    if iters%10==0:
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(transforms.ToPILImage((dummy_data[0].cpu())))

plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')

plt.show()