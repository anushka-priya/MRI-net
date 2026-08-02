import io

import matplotlib.pyplot as plt
import torch
from PIL import Image

from . import config
from .data import get_transforms


def predict_image_bytes(model, image_bytes: bytes, class_names: list, device: torch.device = config.DEVICE) -> dict:
    eval_transform = get_transforms(train=False)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = eval_transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, _ = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())

    return {
        "predicted_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "class_probabilities": {class_names[i]: float(probs[i].item()) for i in range(len(class_names))},
    }


def predict_and_show(model, image_path: str, class_names: list, true_label: str = None,
                      device: torch.device = config.DEVICE) -> dict:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    result = predict_image_bytes(model, image_bytes, class_names, device=device)

    img = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    title = f"Predicted: {result['predicted_class']} ({result['confidence'] * 100:.1f}%)"
    if true_label is not None:
        title += f"\nTrue: {true_label}"
    plt.title(title, color="green" if true_label == result["predicted_class"] else "red")
    plt.axis("off")
    plt.show()
    return result