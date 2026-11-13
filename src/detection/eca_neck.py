
import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer


class ECANeckBlock(nn.Module):
    def __init__(self, block, kernel_size=3):
        super().__init__()
        self.block = block
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Conv1d(
            1, 1, kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        # 2 * sigmoid(0) = 1: preserve initial features.
        nn.init.zeros_(self.channel_conv.weight)

    def forward(self, x):
        x = self.block(x)
        descriptor = self.pool(x).squeeze(-1).transpose(1, 2)
        attention = 2.0 * torch.sigmoid(self.channel_conv(descriptor))
        attention = attention.transpose(1, 2).unsqueeze(-1)
        return x * attention


def add_eca(model):
    if any(isinstance(m, ECANeckBlock) for m in model.modules()):
        return model

    head = model.model[-1]
    indices = list(head.f)
    assert len(indices) == 3, "Expected three detection scales."

    parameter = next(model.parameters())

    for index in indices:
        assert index >= 0
        old = model.model[index]
        assert old.__class__.__name__ == "C2f", (
            f"Unexpected neck block at {index}: {type(old).__name__}"
        )

        new = ECANeckBlock(old).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )

        for attribute in ["i", "f"]:
            setattr(new, attribute, getattr(old, attribute))

        new.type = "ECANeckBlock"
        new.np = sum(p.numel() for p in new.parameters())
        new.train(old.training)
        model.model[index] = new

    return model


class ECATrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(
            cfg=cfg, weights=weights, verbose=verbose
        )
        return add_eca(model)
