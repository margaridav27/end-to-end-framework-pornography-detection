from src.utils.misc import seed

from .vit_lrp import LRP

from typing import Union, Optional
import numpy as np

import torch
import torch.nn as nn


# Fix random seed
seed()


# Source: https://github.com/hila-chefer/Transformer-Explainability/blob/main/Transformer_explainability.ipynb
# Function: Generate Transformer Attribution
def generate_attribution(
    model: nn.Module,
    input: Union[torch.Tensor, np.array],
    label: Optional[Union[torch.Tensor, int]] = None,
    expand_dims: bool = False,
    **kwargs,
):
    device = kwargs.get("device", None)
    if device is None:
        device = next(model.parameters()).device

    if not torch.is_tensor(input):
        input = torch.tensor(input)

    if input.ndim == 3:
        input = input.unsqueeze(0)

    input.requires_grad_()
    input = input.to(device)

    if label is not None and torch.is_tensor(label):
        label = int(label.cpu().item())

    attr = (
        torch.nn.functional.interpolate(
            (
                LRP(model=model)
                .generate_LRP(
                    input=input,
                    index=label,
                    method="transformer_attribution",
                )
                .detach()
            ).reshape(1, 1, 14, 14),
            scale_factor=16,
            mode="bilinear",
        )
        .reshape(224, 224)
        .data.cpu()
        .numpy()
    )

    if expand_dims:
        attr = np.expand_dims(attr, axis=0)

    # Normalization - no need as we are plotting visualizations with Captum
    # attr = (attr - attr.min()) / (attr.max() - attr.min())

    return attr


# Function: Wrapper for compatibility with Quantus
def generate_transformer_explanations(
    model: nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    **kwargs,
):
    return np.array(
        [
            generate_attribution(model, inputs[i], targets[i], **kwargs)
            for i in range(len(targets))
        ]
    )
