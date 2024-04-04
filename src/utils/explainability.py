from src.utils.model import predict
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import os
import gc
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


def set_lrp_rules(model : nn.Module):
    # NOTE: only works with vgg19
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
    norm_mean : List[float],
    norm_std : List[float],
    method_key : str, 
    method_kwargs : Optional[Dict[str, Any]] = None, 
    attribute_kwargs : Optional[Dict[str, Any]] = None, 
    to_explain : Optional[List[str]] = None, 
    batch_size : Optional[int] = 16, 
    noise_tunnel : Optional[bool] = False, 
    noise_tunnel_type : Optional[str] = "SGSQ", 
    noise_tunnel_samples : Optional[int] = 10
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
    if to_explain: 
        for frame_name in to_explain:
            _, input, label = dataset[frame_name]
            input = input.to(device).unsqueeze(0).requires_grad_()
            label = torch.tensor(label).to(device)
            _, pred = predict(model, input)

            if not filter_mask or filter_mask(pred, label):
                print(f"Generating explanations using {ATTRIBUTION_METHODS[method_key][1]} for {frame_name}...")
                if method_key == "LRP-CMP": set_lrp_rules(model)
                attr = method.attribute(inputs=input, target=pred, **attribute_kwargs)
                save_explanation(
                    save_loc=save_loc, 
                    frame=input, 
                    frame_name=frame_name, 
                    norm_mean=norm_mean,
                    norm_std=norm_std,
                    attr=attr[0], 
                    attr_method=method_key if not noise_tunnel else f"{method_key}_NT_{noise_tunnel_type}_{noise_tunnel_samples}", 
                    prediction=pred.item()
                )
    # If to_explain is not specified, generate explanations for the entire test dataset
    else:
        dataloader = DataLoader(dataset, batch_size)
        for names, inputs, labels in dataloader:
            inputs = inputs.to(device).requires_grad_()
            labels = labels.to(device)
            _, preds = predict(model, inputs)

            if filter_mask:
                mask = filter_mask(preds, labels)
                names = [name for name, m in zip(names, mask) if m]
                inputs, preds = inputs[mask], preds[mask]
            
            if len(inputs) == 0: continue 

            print(f"Generating explanations using {ATTRIBUTION_METHODS[method_key][1]} for batch...")
            if method_key == "LRP-CMP": set_lrp_rules(model)
            attrs = method.attribute(inputs=inputs, target=preds, **attribute_kwargs)
            for name, input, pred, attr in zip(names, inputs, preds, attrs):
                save_explanation(
                    save_loc=save_loc, 
                    frame=input, 
                    frame_name=name, 
                    norm_mean=norm_mean,
                    norm_std=norm_std,
                    attr=attr, 
                    attr_method=method_key if not noise_tunnel else f"{method_key}_NT_{noise_tunnel_type}_{noise_tunnel_samples}", 
                    prediction=pred.item()
                )

    # Clear model
    del model

    # Run garbage collector
    gc.collect()


def save_explanation( 
    save_loc : str,
    frame : torch.Tensor,
    frame_name : str,
    norm_mean : List[float],
    norm_std : List[float],
    attr : torch.Tensor,
    attr_method : str,
    prediction : int,
):
    npys_save_loc = os.path.join(save_loc, attr_method, "npys")
    os.makedirs(npys_save_loc, exist_ok=True)

    pngs_save_loc = os.path.join(save_loc, attr_method, "jpgs")
    os.makedirs(pngs_save_loc, exist_ok=True)
    
    print(f"Saving {attr_method} explanation (.npy) for frame {frame_name} to {npys_save_loc}...")
    attribution_np = attr.cpu().detach().numpy()
    np.save(f"{npys_save_loc}/{frame_name}_pred_{prediction}.npy", attribution_np)

    print(f"Saving {attr_method} explanation (.png) for frame {frame_name} to {pngs_save_loc}...")
    fig = visualize_explanation(
        frame=frame,
        frame_name=frame_name,
        norm_mean=norm_mean,
        norm_std=norm_std,
        attr=attribution_np,
        attr_method=attr_method,
        prediction=prediction,
        side_by_side=True,
        colormap="jet"
    )
    fig.savefig(f"{pngs_save_loc}/{frame_name}_pred_{prediction}.png")
    plt.close(fig)


def visualize_explanation(
    frame,
    frame_name : str,
    norm_mean : List[float],
    norm_std : List[float],
    attr : Union[np.ndarray, str],
    attr_method : str,
    prediction : int,
    sign : Optional[str] = "positive",
    vis_method : Optional[str] = "blended_heat_map",
    side_by_side : Optional[bool] = False,
    colormap : Optional[str] = None,
    outlier_perc : Optional[Union[float, int]] = 2,
    alpha_overlay : Optional[float] = 0.5 
):
    base_attr_method = attr_method.split("_")[0]
    noise_tunnel = True if len(attr_method.split("_")) > 1 else False

    title_original_image = f"{frame_name} (pred: {prediction})"
    title_method = f"{ATTRIBUTION_METHODS[base_attr_method][1]}"
    if noise_tunnel:
        noise_tunnel_type = NOISE_TUNNEL_TYPES[attr_method.split("_")[-2]]
        noise_tunnel_samples = attr_method.split("_")[-1]
        title_method += f" with Noise Tunnel ({noise_tunnel_type}, {noise_tunnel_samples} samples)"

    if isinstance(attr, str): attr = np.load(attr)
    if len(attr.shape) == 4: attr = attr[0]
    attr = np.transpose(attr, (1,2,0))
    np_frame = frame.squeeze().cpu().permute(1,2,0).detach().numpy()

    if side_by_side:
        # Denormalize for better visualization
        np_frame = np.clip(norm_std * np_frame + norm_mean, 0, 1)

        return viz.visualize_image_attr_multiple(
            attr=attr,
            original_image=np_frame,
            methods=["original_image", vis_method],
            signs=["all", sign],
            show_colorbar=True,
            cmap=colormap,
            outlier_perc=outlier_perc,
            alpha_overlay=alpha_overlay,
            titles=[title_original_image, title_method]
        )[0]
    else:
        return viz.visualize_image_attr(
            attr=attr,
            original_image=np_frame,
            method=vis_method,
            sign=sign,
            show_colorbar=True,
            cmap=colormap,
            outlier_perc=outlier_perc,
            alpha_overlay=alpha_overlay,
            title=f"{title_original_image} - {title_method}"
        )[0]
