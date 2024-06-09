from typing import Optional, Union, Dict, Any
import numpy as np

import torch
import torch.nn as nn

from captum.attr import (
    IntegratedGradients,
    DeepLift,
    LRP,
    Deconvolution,
    Occlusion,
    NoiseTunnel,
)
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule


NOISE_TUNNEL_TYPES = {"SG": "smoothgrad", "SGSQ": "smoothgrad_sq", "VG": "vargrad"}

ATTRIBUTION_METHODS = {
    "IG": IntegratedGradients,
    "DEEP-LIFT": DeepLift,
    "LRP": LRP,
    "LRP-CMP": LRP,
    "DECONV": Deconvolution,
    "OCC": Occlusion,
}


# FIXME: only works with vgg19
def _set_lrp_rules(model: nn.Module):
    layers = list(model.module.features) + list(model.module.classifier)
    num_layers = len(layers)

    for idx_layer in range(1, num_layers):
        if idx_layer <= 20:
            setattr(layers[idx_layer], "rule", GammaRule())
        elif 21 <= idx_layer <= 36:
            setattr(layers[idx_layer], "rule", EpsilonRule(epsilon=0.25))
        elif idx_layer >= 37:
            setattr(layers[idx_layer], "rule", EpsilonRule(epsilon=0))


def get_captum_method_name(
    method_name: str,
    noise_tunnel: bool = False,
    noise_tunnel_type: Optional[str] = None,
    noise_tunnel_samples: Optional[int] = None,
):
    name = f"{method_name}"
    if noise_tunnel:
        assert noise_tunnel_type is not None and noise_tunnel_samples is not None
        name += f"_NT_{noise_tunnel_type}_{noise_tunnel_samples}"
    return name


def generate_captum_explanations(
    model: nn.Module,
    inputs: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    method_name: str,
    method_kwargs: Optional[Dict[str, Any]] = None,
    attribute_kwargs: Optional[Dict[str, Any]] = None,
    noise_tunnel: bool = False,
    noise_tunnel_type: str = "SGSQ",
    noise_tunnel_samples: int = 10,
    reduce_channels: bool = False,
    device: Optional[str] = None,
) -> np.ndarray:
    if method_name not in ATTRIBUTION_METHODS.keys():
        raise ValueError(f"Invalid Captum method: {method_name}")
    if noise_tunnel and noise_tunnel_type not in NOISE_TUNNEL_TYPES.keys():
        raise ValueError(f"Invalid Captum noise tunnel type: {noise_tunnel_type}")

    # Initialize attribution method's kwargs
    if method_kwargs is None:
        method_kwargs = {}
    if attribute_kwargs is None:
        attribute_kwargs = {}

    if not torch.is_tensor(inputs):
        inputs = torch.tensor(inputs)
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)

    # Add batch dimension
    if inputs.ndim == 3:
        inputs = inputs.unsqueeze(0)

    inputs = inputs.to(device)
    targets = targets.to(device)

    # Set model to evaluation mode
    model.to(device)
    model.eval()

    # Initialize attribution method
    method = ATTRIBUTION_METHODS[method_name](model, **method_kwargs)

    if noise_tunnel:
        method = NoiseTunnel(method)
        attribute_kwargs["nt_type"] = NOISE_TUNNEL_TYPES[noise_tunnel_type]
        attribute_kwargs["nt_samples"] = noise_tunnel_samples

    if method_name == "LRP-CMP":
        _set_lrp_rules(model)

    # Perform attribution to batch
    attributions = method.attribute(inputs=inputs, target=targets, **attribute_kwargs)
    
    if torch.is_tensor(attributions):
        if attributions.requires_grad:
            attributions = attributions.detach().cpu().numpy()
        else:
            attributions = attributions.cpu().numpy()

    # Reduce channels dimension to 1: BCHW -> B1HW
    if reduce_channels:
        attributions = np.sum(attributions, axis=1, keepdims=True)

    return attributions
