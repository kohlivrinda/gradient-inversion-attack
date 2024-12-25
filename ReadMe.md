# Gradient Inversion Attack: Deep Leakage from Gradients (DLG)

This repository implements the **Deep Leakage from Gradients (DLG)** attack, demonstrating how sensitive training data can be reconstructed from gradients shared during Federated Learning (FL).

---

## 🛠 Features
- Supports **LeNet** and **ResNet-18** architectures for the attack on CIFAR-10.
- CLI-based workflow for easy experimentation.
- Generates intermediate and final reconstructed images for visualization.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- PyTorch
- torchvision
- matplotlib

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/gradient-inversion-attack.git
   cd gradient-inversion-attack
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔧 Running the Attack

Use `main.py` to run the attack via the command line. The script supports the following arguments:

### CLI Arguments
| Argument      | Type    | Default | Description                                           |
|---------------|---------|---------|-------------------------------------------------------|
| `--model`     | string  | None    | Specify the model: `lenet` or `resnet`. (Required)    |
| `--img_idx`   | integer | 25      | Index of the CIFAR-10 image to attack.                |
| `--epochs`    | integer | 500     | Number of optimization iterations for the attack.     |

### Example Command
To run the attack on LeNet for 500 epochs:
```bash
python main.py --model lenet --img_idx 25 --epochs 500
```

To switch to ResNet:
```bash
python main.py --model resnet --img_idx 30 --epochs 1000
```

---

## 📊 Outputs
1. **Ground Truth Image:** The original CIFAR-10 image selected for the attack, saved as `gt_image.png` in the `plots/` directory.
2. **Dummy Image Initialization:** Randomly initialized dummy image (`dummy_img.png`).
3. **Reconstructed Images:** Saved progressively during the attack (`reconstruction.png`), visualizing the attack's performance.

---

## 🔍 Observations
- **LeNet:** Successfully reconstructs images in ~300-500 iterations, with results closely matching the ground truth.
- **ResNet-18:** Reconstruction is more challenging due to the complexity of the model and gradient dispersion but remains feasible under certain conditions.

---

## 🌱 Next Steps
- Test the attack on deeper models like ResNet-56.
- Experiment with mitigation strategies such as gradient perturbation and compression.
- Extend support for other datasets (e.g., MNIST, ImageNet).
- Understand what works and what doesn't when dealing with deeper models.

---

## 📚 References
- Original Paper: [Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935)
- Project Insights: Implementation of the DLG attack on CIFAR-10 using LeNet and ResNet.

---

Star the repo if you find it helpful! 🌟