from src.utils.misc import seed

from .vit_lrp import LRP

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch


# Fix random seed
seed()


# Function: Function to unnormalize images
def unnormalize(image, mean_array, std_array):

    # Create a copy
    unnormalized_img = image.copy()

    # Get channels
    _, _, channels = unnormalized_img.shape

    for c in range(channels):
        unnormalized_img[:, :, c] = image[:, :, c] * std_array[c] + mean_array[c]

    return unnormalized_img


# Helper funtion to get figures to be shown after Captum VIZ
# https://stackoverflow.com/questions/49503869/attributeerror-while-trying-to-load-the-pickled-matplotlib-figure
def convert_figure(fig):

    # create a dummy figure and use its manager to display "fig"
    dummy = plt.figure(figsize=(6, 6))
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)


# Source: https://github.com/hila-chefer/Transformer-Explainability/blob/main/Transformer_explainability.ipynb
# Function: Generate transformer attribution
def generate_attribution(
    model,
    image,
    label=None,
    **kwargs,
):
    device = next(model.parameters()).device

    unnormalized_image = unnormalize(
        image=np.transpose(image.cpu().numpy(), (1, 2, 0)),
        mean_array=kwargs["mean_array"],
        std_array=kwargs["std_array"],
    )

    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image.requires_grad = True
    image = image.to(device)

    if label is not None and torch.is_tensor(label):
        label = int(label.cpu().item())

    attr = (
        torch.nn.functional.interpolate(
            (
                LRP(model=model)
                .generate_LRP(
                    input=image,
                    method="transformer_attribution",
                    index=label,
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

    # Normalization
    attr = (attr - attr.min()) / (attr.max() - attr.min())

    return unnormalized_image, attr


# Source: https://github.com/hila-chefer/Transformer-Explainability/blob/main/Transformer_explainability.ipynb
# Function: Create overlay of attributions on image
def overlay(image, attr):
    # Apply JET colormap
    heatmap = cv2.applyColorMap(np.uint8(attr * 255), cv2.COLORMAP_JET)

    # Combine image with heatmap
    overlay = (np.float32(heatmap) / 255) + np.float32(image)
    overlay = overlay / np.max(overlay)

    return overlay


# Source: https://github.com/hila-chefer/Transformer-Explainability/blob/main/Transformer_explainability.ipynb
# Function: Generate visualization of transformer attributions
def generate_attribution_visualization(image, attr):
    vis = overlay(image, attr)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis
