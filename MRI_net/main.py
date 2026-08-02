import argparse

import torch
import uvicorn
from torch.utils.data import DataLoader

from . import config
from .data import build_dataloaders, build_datasets, check_train_test_overlap, get_class_names
from .evaluation import evaluate_classification, plot_training_curves, visualize_reconstructions
from .model import build_model
from .train import train_model


def run_train() -> None:
    torch.manual_seed(config.SEED)

    class_names = get_class_names()
    print("Classes:", class_names)

    check_train_test_overlap()

    train_dataset, test_dataset = build_datasets(class_names)
    train_loader, test_loader = build_dataloaders(train_dataset, test_dataset)
    print(f"Train: {len(train_dataset)} images | Test: {len(test_dataset)} images")

    model = build_model(num_classes=len(class_names))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    history = train_model(model, train_loader, test_loader, class_names)

    plot_training_curves(history)
    evaluate_classification(model, test_loader, class_names)
    visualize_reconstructions(model, test_loader, class_names)


def run_evaluate() -> None:
    class_names = get_class_names()
    _, test_dataset = build_datasets(class_names)
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS
    )

    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    model = build_model(num_classes=len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state"])

    evaluate_classification(model, test_loader, checkpoint["class_names"])
    visualize_reconstructions(model, test_loader, checkpoint["class_names"])


def run_serve() -> None:
    uvicorn.run(
        "mri_net.api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Brain MRI tumor classifier pipeline.")
    parser.add_argument("command", choices=["train", "evaluate", "serve"])
    args = parser.parse_args()

    if args.command == "train":
        run_train()
    elif args.command == "evaluate":
        run_evaluate()
    elif args.command == "serve":
        run_serve()


if __name__ == "__main__":
    main()