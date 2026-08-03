import inspect

import torch
from torch.utils.data import Dataset, DataLoader

from MRI_net.data import get_class_names, get_transforms
from MRI_net.model import build_model
from MRI_net.train import train_model


class TinyFakeDataset(Dataset):
    def __init__(self, n=4, num_classes=4):
        self.n = n
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.randn(3, 128, 128), torch.randint(0, self.num_classes, (1,)).item()


def test_bug_1():
    source = inspect.getsource(get_class_names)
    assert "sorted(" in source, "test failed"


def test_bug_2():
    model = build_model(num_classes=4, device=torch.device("cpu"))
    recon_params_before = [p.clone() for p in model.reconstructor.parameters()]

    train_ds = TinyFakeDataset(4)
    test_ds = TinyFakeDataset(2)
    train_loader = DataLoader(train_ds, batch_size=2)
    test_loader = DataLoader(test_ds, batch_size=2)

    train_model(
        model, train_loader, test_loader,
        class_names=["a", "b", "c", "d"],
        device=torch.device("cpu"),
        epochs=1,
        checkpoint_path="tests_tmp_checkpoint.pth",
    )

    recon_params_after = list(model.reconstructor.parameters())
    changed = any(
        not torch.equal(before, after)
        for before, after in zip(recon_params_before, recon_params_after)
    )
    assert changed, "test failed"


def test_bug_3():
    train_tf = str(get_transforms(train=True))
    eval_tf = str(get_transforms(train=False))
    assert train_tf != eval_tf, "test failed"