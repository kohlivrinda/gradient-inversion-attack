import torch
import torch.nn.functional as F

def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

# For image labels, instead of directly optimizing the discrete categorical values,
# we random initialize a vector with shape N × C
# where N is the batch size and C is the number of classes,
# and then take its softmax output as the one-hot label for optimization.