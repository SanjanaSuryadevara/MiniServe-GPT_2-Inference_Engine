"""Weight-only int8 quantization (symmetric, per output channel)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    """Stores weights as int8 plus one fp scale per output row (about 4x smaller than fp32).

    y = (x @ Wq^T) * scale + b, which equals x @ (Wq * scale)^T + b, so the scaled weight
    matrix is never materialised.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.data
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        self.register_buffer("qweight", torch.round(w / scale).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale.squeeze(1).clone())
        self.register_buffer("bias", linear.bias.data.clone() if linear.bias is not None else None)

    def forward(self, x):
        y = F.linear(x, self.qweight.to(x.dtype)) * self.scale
        return y if self.bias is None else y + self.bias


def quantize_model(model: nn.Module) -> nn.Module:
    """Replace every nn.Linear inside the transformer blocks with Int8Linear (in place).
    Embeddings and LayerNorms stay in full precision."""
    for block in model.blocks:
        for parent in block.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, Int8Linear(child))
    return model


def model_size_bytes(model: nn.Module) -> int:
    tensors = list(model.parameters()) + list(model.buffers())
    return sum(t.numel() * t.element_size() for t in tensors)
