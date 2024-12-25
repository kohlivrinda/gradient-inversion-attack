import argparse
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from src.utils import label_to_onehot, cross_entropy_for_onehot
from src.model import ModifiedResnet18, LeNet
from src.attack import run_attack 

# CLI Argument Parser
def get_args():
    parser = argparse.ArgumentParser(description="Run Gradient Inversion Attack")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["lenet", "resnet"], 
        required=True, 
        help="Specify the model: lenet or resnet"
    )
    parser.add_argument(
        "--img_idx", 
        type=int, 
        default=25, 
        help="Index of the image to attack (default: 25)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=500, 
        help="Number of optimization iterations for the attack (default: 500)"
    )
    return parser.parse_args()

def main():
    args = get_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = datasets.CIFAR10(root='../data', download=True)
    data_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor()
    ])
    to_pil_image = transforms.ToPILImage()
    
    # Select model
    if args.model == "lenet":
        model = LeNet().to(device)
    elif args.model == "resnet":
        model = ModifiedResnet18().to(device)
    
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    model.apply(weights_init)
    
    # Select image and label
    gt_img = data_transforms(dataset[args.img_idx][0]).to(device) #(C,H,W)
    gt_img = gt_img.view(1, *gt_img.size()) #(N,C,H,W)
    gt_label = torch.Tensor([dataset[args.img_idx][1]]).long().to(device) #scalar
    gt_label = gt_label.view(1, )  # (N, ) expected by loss function
    gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

    
    # Visualize ground truth image
    plt.imshow(to_pil_image(gt_img[0].cpu()))
    plt.title("Ground Truth Image")
    plt.savefig('plots/gt_image.png', format='png', dpi=300)
    plt.show()

    # Run the attack
    run_attack(model, gt_img, gt_onehot_label, device, to_pil_image, args.epochs)

if __name__ == "__main__":
    main()
