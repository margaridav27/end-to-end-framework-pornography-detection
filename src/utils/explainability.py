from src.utils.misc import unnormalize
from src.utils.model import predict
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from captum.attr import (
  visualization as viz, 
  IntegratedGradients, 
  DeepLift, 
  LRP,
  Deconvolution, 
  Occlusion, 
  NoiseTunnel,
)
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule


NOISE_TUNNEL_TYPES = { 
    "SG": "smoothgrad", 
    "SGSQ": "smoothgrad_sq", 
    "VG": "vargrad" 
}

ATTRIBUTION_METHODS = {
    "IG": (IntegratedGradients, "Integrated Gradients"),
    "DEEP-LIFT": (DeepLift, "Deep Lift"),
    "LRP": (LRP, "LRP"),
    "LRP-CMP": (LRP, "LRP (composite strategy)"),
    "DECONV": (Deconvolution, "Deconvolution"),
    "OCC": (Occlusion, "Occlusion")
}

VISUALIZATION_TYPES = { "heat_map", "blended_heat_map", "masked_image", "alpha_scaling" } 

SIGN_TYPES = { "all", "positive", "negative", "absolute_value" }


# FIXME: only works with vgg19
def set_lrp_rules(model : nn.Module):
    layers = list(model.module.features) + list(model.module.classifier)
    num_layers = len(layers)

    for idx_layer in range(1, num_layers):
        if idx_layer <= 20:
            setattr(layers[idx_layer], "rule", GammaRule())
        elif 21 <= idx_layer <= 36:
            setattr(layers[idx_layer], "rule", EpsilonRule(epsilon=0.25))
        elif idx_layer >= 37:
            setattr(layers[idx_layer], "rule", EpsilonRule(epsilon=0))


def generate_explanations(
    save_loc : str,
    model : nn.Module,
    filter : str, 
    device : str,
    dataset : PornographyFrameDataset, 
    method_key : str, 
    method_kwargs : Optional[Dict[str, Any]] = None, 
    attribute_kwargs : Optional[Dict[str, Any]] = None, 
    noise_tunnel : bool = False, 
    noise_tunnel_type : str = "SGSQ", 
    noise_tunnel_samples : int = 10,
    to_explain : Optional[List[str]] = None, 
    batch_size : int = 16, 
    **kwargs
):
    if not method_kwargs: method_kwargs = {}
    if not attribute_kwargs: attribute_kwargs = {}

    method = ATTRIBUTION_METHODS[method_key][0](model, **method_kwargs)
    if noise_tunnel:
        method = NoiseTunnel(method)
        attribute_kwargs["nt_type"] = NOISE_TUNNEL_TYPES[noise_tunnel_type]
        attribute_kwargs["nt_samples"] = noise_tunnel_samples

    # Define filter mask based on filter
    filter_mask = None
    if filter == "correct":
        filter_mask = lambda preds, labels: preds == labels
    elif filter == "incorrect":
        filter_mask = lambda preds, labels: preds != labels

    # If to_explain is specified, generate explanations for those frames
    # No filter is applied in this case
    if to_explain: 
        for frame_name in to_explain:
            _, input, label, _ = dataset[frame_name]
            input = input.to(device).unsqueeze(0).requires_grad_()
            _, pred = predict(model, input)

            if method_key == "LRP-CMP": set_lrp_rules(model)
            attr = method.attribute(inputs=input, target=pred, **attribute_kwargs)[0]
            save_explanation(
                save_loc=os.path.join(
                    save_loc,
                    "correct" if pred.item() == label else "incorrect",
                    (
                        method_key
                        if not noise_tunnel
                        else f"{method_key}_NT_{noise_tunnel_type}_{noise_tunnel_samples}"
                    ),
                ),
                image=input.squeeze().cpu().permute(1, 2, 0).detach().numpy(),
                image_name=frame_name,
                attr=attr,
                **kwargs,
            )
    # If to_explain is not specified, generate explanations for the entire dataset
    else:
        dataloader = DataLoader(dataset, batch_size)
        for names, inputs, labels, _ in dataloader:
            inputs = inputs.to(device).requires_grad_()
            labels = labels.to(device)
            _, preds = predict(model, inputs)

            if filter_mask:
                mask = filter_mask(preds, labels)
                names = [name for name, m in zip(names, mask) if m]
                inputs, preds = inputs[mask], preds[mask]

            if len(inputs) == 0: continue 

            if method_key == "LRP-CMP": set_lrp_rules(model)
            attrs = method.attribute(inputs=inputs, target=preds, **attribute_kwargs)
            for name, input, attr in zip(names, inputs, attrs):
                save_explanation(
                    save_loc=os.path.join(save_loc, filter, method_key if not noise_tunnel else f"{method_key}_NT_{noise_tunnel_type}_{noise_tunnel_samples}"), 
                    image=input.squeeze().cpu().permute(1,2,0).detach().numpy(), 
                    image_name=name, 
                    attr=attr, 
                    **kwargs
                )


def save_explanation( 
    save_loc : str,
    image : Union[torch.Tensor, np.ndarray],
    image_name : str,
    attr : Union[torch.Tensor, np.ndarray],
    **kwargs
):
    npys_save_loc = os.path.join(save_loc, "npys")
    os.makedirs(npys_save_loc, exist_ok=True)
    if torch.is_tensor(attr): attr = attr.cpu.numpy()
    np.save(f"{npys_save_loc}/{os.path.splitext(image_name)[0]}.npy", attr)

    jpgs_save_loc = os.path.join(save_loc, "jpgs")
    os.makedirs(jpgs_save_loc, exist_ok=True)
    fig = visualize_explanation(
        image=image,
        attr=attr,
        **kwargs
    )
    fig.savefig(f"{jpgs_save_loc}/{image_name}")
    plt.close(fig)


def visualize_explanation(
    image: Union[torch.Tensor, np.ndarray],
    attr: Union[np.ndarray, str],
    sign: str = "positive",
    method: str = "blended_heat_map",
    colormap: Optional[str] = None,
    outlier_perc: Union[float, int] = 2,
    alpha_overlay: float = 0.5,
    show_colorbar: bool = False,
    side_by_side: bool = False,
    **kwargs,
):
    CHW = lambda shape: shape[0] in (3, 4)
    HWC = lambda input: np.transpose(input, (1, 2, 0))

    # Convert attr to numpy array of shape (H, W, C)
    if isinstance(attr, str): attr = np.load(attr)
    if len(attr.shape) == 4: attr = np.squeeze(attr)
    if len(attr.shape) == 2: attr = np.expand_dims(attr, axis=-1)
    if CHW(attr.shape): attr = HWC(attr)
    
    # Convert image to numpy array of shape (H, W, C)
    if torch.is_tensor(image): image = image.cpu().numpy()
    if len(image.shape) == 4: image = np.squeeze(image)
    if CHW(image.shape): image = HWC(image)

    assert image.shape[:2] == attr.shape[:2], "Image and attr shapes must match."

    if side_by_side:
        norm_std = kwargs.get("norm_std", None)
        norm_mean = kwargs.get("norm_mean", None)

        # Unnormalize for better visualization
        if norm_std and norm_mean:
            image = unnormalize(image, norm_mean, norm_std)

        return viz.visualize_image_attr_multiple(
            attr=attr,
            original_image=image,
            methods=["original_image", method],
            signs=["all", sign],
            cmap=colormap,
            show_colorbar=show_colorbar,
            outlier_perc=outlier_perc,
            alpha_overlay=alpha_overlay,
        )[0]
    else:
        return viz.visualize_image_attr(
            attr=attr,
            original_image=image,
            method=method,
            sign=sign,
            cmap=colormap,
            show_colorbar=show_colorbar,
            outlier_perc=outlier_perc,
            alpha_overlay=alpha_overlay,
        )[0]
