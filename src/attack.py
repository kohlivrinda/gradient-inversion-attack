import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from utils import label_to_onehot, cross_entropy_for_onehot
from model import ModifiedResnet18, LeNet
import matplotlib.pyplot as plt

torch.manual_seed(50)

dataset = datasets.CIFAR10(root='../data', download = True)
data_transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor()
    ])

to_pil_image = transforms.ToPILImage()
device = "cuda" if torch.cuda.is_available() else 'cpu'

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)
    

# model = ModifiedResnet18().to(device)
model = LeNet().to(device) 
model.apply(weights_init)


criterion = cross_entropy_for_onehot
img_idx = 25

gt_img = data_transforms(dataset[img_idx][0]).to(device) #(C,H,W)
gt_img = gt_img.view(1, *gt_img.size()) #(N,C,H,W)
gt_label = torch.Tensor([dataset[img_idx][1]]).long().to(device) # scalar 
gt_label = gt_label.view(1, ) # (N, ) expected by loss function
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

plt.imshow(to_pil_image(gt_img[0].cpu()))
plt.savefig('../plots/gt_image1.png', format='png', dpi=300)

# original gradient calculation
pred = model(gt_img)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, model.parameters())

#share gradient with other clients
og_dy_dx = list((_.detach().clone() for _ in dy_dx))

# create dummy data
dummy_data = torch.randn(gt_img.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

plt.imshow(to_pil_image(dummy_data[0].cpu()))
plt.savefig('../plots/dummy_img1.png', format='png', dpi=300)

#DLG Algo:

opt = torch.optim.LBFGS([dummy_data, dummy_label])
history = []

for iters in range(500):
    def closure():
        opt.zero_grad()
        dummy_pred = model(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

        grad_diff = 0
        grad_count = 0
        for gx, gy in zip(dummy_dy_dx, og_dy_dx):
            grad_diff += ((gx-gy)**2).sum()
            grad_count += gx.nelement()
        grad_diff = grad_diff / grad_count * 1000
        grad_diff.backward()
        return grad_diff
    
    opt.step(closure)

    if iters%10==0:
        current_loss = closure()
        print(f" Inversion loss at iter {iters}: {current_loss.item()}")
        history.append(to_pil_image(dummy_data[0].cpu()))

plt.figure(figsize=(12, 11))
for i in range(50):
    plt.subplot(5, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')
    plt.savefig('../plots/reconstruction2.png', format='png')

plt.show()