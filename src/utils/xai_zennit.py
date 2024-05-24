from typing import Optional, Union, Dict, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import zennit.composites as composites
from zennit.composites import *

import zennit.attribution as attributors
from zennit.attribution import *


def get_zennit_method_name(method_name: str, composite_name: Optional[str] = None):
    name = f"{method_name}"
    if composite_name:
        name += f"_{composite_name}"
    return name


def generate_zennit_explanations(
    model: nn.Module,
    inputs: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    method_name: str,
    method_kwargs: Optional[Dict[str, Any]] = None,
    composite_name: Optional[str] = None,
    composite_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    if not torch.is_tensor(inputs):
        inputs = torch.tensor(inputs)
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)

    # Add batch dimension
    if inputs.ndim == 3:
        inputs = inputs.unsqueeze(0)

    targets = F.one_hot(targets, num_classes=2)

    inputs.to(device)
    targets.to(device)

    # Set model to evaluation mode
    model.to(device)
    model.eval()

    composite = None
    if composite_name:
        composite = getattr(composites, composite_name, None)
        if composite is None:
            raise ValueError(f"Invalid Zennit composite: {composite_name}")

        if not composite_kwargs:
            composite_kwargs = {}

        try:
            composite = composite(**composite_kwargs)
        except:
            raise AssertionError(f"Invalid parameter for Zennit composite: {composite_kwargs}")

    method = getattr(attributors, method_name, None)
    if method is None:
        raise ValueError(f"Invalid Zennit attributor: {method_name}")

    if not method_kwargs:
        method_kwargs = {}

    try:
        with method(model=model, composite=composite if composite else None, **method_kwargs) as attributor:
            _, attributions = attributor(inputs, targets)
        if attributions.requires_grad:
            return attributions.detach().cpu().numpy()
        else:
            return attributions.cpu().numpy()
    except:
        raise AssertionError(f"Invalid parameter for Zennit attributor: {method_kwargs}")
