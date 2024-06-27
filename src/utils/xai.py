from src.utils.model import predict
from src.utils.xai_captum import generate_captum_explanations, get_captum_method_name
from src.utils.xai_zennit import generate_zennit_explanations, get_zennit_method_name
from src.utils.xai_visualization import visualize_explanation
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import os
from typing import Dict, List, Optional, Any, Union
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def generate_explanations(
    save_loc: str,
    model: nn.Module,
    dataset: PornographyFrameDataset,
    method_cfg: Dict[str, Any],
    library: str = "captum",
    filter: str = "correct",
    to_explain: Optional[List[str]] = None,
    batch_size: int = 16,
    device: Optional[str] = None,
    **kwargs,
):
    generator = None
    generator_kwargs = {}

    if library == "captum":
        generator = generate_captum_explanations
        generator_kwargs["method_name"] = method_cfg.get("method_name", None)
        generator_kwargs["method_kwargs"] = method_cfg.get("method_kwargs", {})
        generator_kwargs["attribute_kwargs"] = method_cfg.get("attribute_kwargs", {})
        generator_kwargs["noise_tunnel"] = method_cfg.get("noise_tunnel", False)
        generator_kwargs["noise_tunnel_type"] = method_cfg.get("noise_tunnel_type", None)
        generator_kwargs["noise_tunnel_samples"] = method_cfg.get("noise_tunnel_samples", None)
    elif library == "zennit":
        generator = generate_zennit_explanations
        generator_kwargs["method_name"] = method_cfg.get("method_name", None)
        generator_kwargs["method_kwargs"] = method_cfg.get("method_kwargs", {})
        generator_kwargs["composite_name"] = method_cfg.get("composite_name", None)
        generator_kwargs["composite_kwargs"] = method_cfg.get("composite_kwargs", {})
        generator_kwargs["canonizer_name"] = method_cfg.get("canonizer_name", None)

    if not generator_kwargs["method_name"]:
        raise ValueError("No method was provided to generate the attributions")

    method_name = (
        get_captum_method_name(
            generator_kwargs["method_name"],
            generator_kwargs["noise_tunnel"],
            generator_kwargs["noise_tunnel_type"],
            generator_kwargs["noise_tunnel_samples"],
        )
        if library == "captum"
        else get_zennit_method_name(
            generator_kwargs["method_name"], 
            generator_kwargs["composite_name"]
        )
    )

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
            input = input.to(device).unsqueeze(0)
            _, pred = predict(model, input)

            attr = generator(
                model=model,
                inputs=input,
                targets=pred,
                device=device,
                **generator_kwargs,
            )[0]
            save_explanation(
                save_loc=os.path.join(
                    save_loc,
                    "correct" if pred.item() == label else "incorrect",
                    method_name,
                ),
                image=input,
                image_name=frame_name,
                attr=attr,
                **kwargs,
            )
    # If to_explain is not specified, generate explanations for the entire dataset (filtered, if any is specified)
    else:
        dataloader = DataLoader(dataset, batch_size)
        for names, inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            _, preds = predict(model, inputs)

            if filter_mask:
                mask = filter_mask(preds, labels)
                names = [name for name, m in zip(names, mask) if m]
                inputs, preds = inputs[mask], preds[mask]

            if len(inputs) == 0:
                continue

            attrs = generator(
                model=model,
                inputs=inputs,
                targets=preds,
                device=device,
                **generator_kwargs,
            )
            for name, input, attr in zip(names, inputs, attrs):
                save_explanation(
                    save_loc=os.path.join(
                        save_loc,
                        filter,
                        method_name,
                    ),
                    image=input,
                    image_name=name,
                    attr=attr,
                    **kwargs,
                )


def save_explanation(
    save_loc: str,
    image: Union[torch.Tensor, np.ndarray],
    image_name: str,
    attr: Union[torch.Tensor, np.ndarray],
    **kwargs,
):
    npys_save_loc = os.path.join(save_loc, "npys")
    os.makedirs(npys_save_loc, exist_ok=True)
    if torch.is_tensor(attr):
        attr = attr.cpu.numpy()
    np.save(f"{npys_save_loc}/{os.path.splitext(image_name)[0]}.npy", attr)

    jpgs_save_loc = os.path.join(save_loc, "jpgs")
    os.makedirs(jpgs_save_loc, exist_ok=True)
    fig = visualize_explanation(image=image, attr=attr, **kwargs)
    fig.savefig(f"{jpgs_save_loc}/{image_name}")
    plt.close(fig)
