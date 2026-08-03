import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from . import config


def plot_training_curves(history: dict, output_dir: str = config.ARTIFACTS_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_cls_loss"], label="cls_loss")
    axes[0].plot(history["train_recon_loss"], label="recon_loss")
    axes[0].set_title("Training Losses")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train_acc")
    axes[1].plot(history["val_acc"], label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(output_path, dpi=150)
    plt.show()
    return output_path


def evaluate_classification(model, test_loader, class_names: list, device: torch.device = config.DEVICE,
                             output_dir: str = config.ARTIFACTS_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits, _ = model(images)
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix -- Multi-Task ResNet-from-Scratch")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path, dpi=150)
    plt.show()
    return output_path


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 2)


def visualize_reconstructions(model, test_loader, class_names: list, device: torch.device = config.DEVICE,
                               num_samples: int = 5, output_dir: str = config.ARTIFACTS_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    images, labels = next(iter(test_loader))
    images = images.to(device)
    with torch.no_grad():
        _, recon = model(images)

    fig, axes = plt.subplots(2, num_samples, figsize=(14, 6))
    for i in range(num_samples):
        axes[0, i].imshow(denormalize(images[i]).permute(1, 2, 0))
        axes[0, i].set_title(f"Original ({class_names[labels[i]]})")
        axes[0, i].axis("off")
        axes[1, i].imshow(denormalize(recon[i]).permute(1, 2, 0))
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis("off")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "reconstructions.png")
    plt.savefig(output_path, dpi=150)
    plt.show()
    return output_path
