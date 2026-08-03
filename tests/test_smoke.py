import torch
from MRI_net.model import build_model


def test_model_forward_runs():
    model = build_model(num_classes=4, device=torch.device("cpu"))
    x = torch.randn(2, 3, 128, 128)
    logits, recon = model(x)
    assert logits.shape == (2, 4)
    assert recon.shape[0] == 2