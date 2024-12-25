import torch
import matplotlib.pyplot as plt
from src.utils import cross_entropy_for_onehot
import torch.nn.functional as F

def run_attack(model, gt_img, gt_onehot_label, device, to_pil_image, epochs=500):
    """
    Runs the gradient inversion attack.

    Args:
        model: PyTorch model to attack (e.g., LeNet, ResNet).
        gt_img: Ground truth image tensor (N, C, H, W).
        gt_onehot_label: One-hot encoded ground truth label tensor (N, num_classes).
        device: 'cuda' or 'cpu'.
        to_pil_image: tensor -> PIL image transform.
        epochs: num(iter) (default: 500).
    """
    model.eval()

    # calc original gradients
    pred = model(gt_img)
    criterion = cross_entropy_for_onehot
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, model.parameters())
    og_dy_dx = [_.detach().clone() for _ in dy_dx]

    # init dummy data and labels
    dummy_data = torch.randn(gt_img.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

    plt.imshow(to_pil_image(dummy_data[0].cpu()))
    plt.title("Initial Dummy Image")
    plt.savefig('plots/dummy_img.png', format='png', dpi=300)
    plt.show()

    opt = torch.optim.LBFGS([dummy_data, dummy_label])
    history = []

    # DLG 
    for iters in range(epochs):
        def closure():
            opt.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            grad_count = 0
            for gx, gy in zip(dummy_dy_dx, og_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
                grad_count += gx.nelement()
            grad_diff = grad_diff / grad_count * 1000
            grad_diff.backward()
            return grad_diff

        opt.step(closure)

        if iters % 10 == 0:
            current_loss = closure()
            print(f"Inversion loss at iter {iters}: {current_loss.item()}")
            history.append(to_pil_image(dummy_data[0].cpu()))

    # save reconstruction results
    plt.figure(figsize=(12, 11))
    for i in range(min(len(history), 50)):
        plt.subplot(5, 10, i + 1)
        plt.imshow(history[i])
        plt.title(f"iter={i * 10}")
        plt.axis('off')
    plt.savefig('plots/reconstruction.png', format='png', dpi=300)
    plt.show()
