from src.utils.misc import seed

from .vit_lrp import LRP

import torch


# Fix random seed
seed()


# Source: https://github.com/hila-chefer/Transformer-Explainability/blob/main/Transformer_explainability.ipynb
# Function: Generate transformer attribution
def generate_attribution(
    model,
    image,
    label=None
):
    device = next(model.parameters()).device

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

    # Normalization - no need as we are plotting visualizations with Captum
    # attr = (attr - attr.min()) / (attr.max() - attr.min())

    return attr
