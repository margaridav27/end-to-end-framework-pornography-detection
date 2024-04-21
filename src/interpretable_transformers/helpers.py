# Source: https://github.com/hila-chefer/Transformer-Explainability
# Model creation / weight loading / state_dict helpers
# Hacked together by / Copyright 2020 Ross Wightman

import timm


def load_timm_model(model_name: str, pretrained: bool = True, num_classes: int = 2):
    model = timm.create_model(
        model_name=model_name, 
        pretrained=pretrained, 
        num_classes=num_classes
    )

    return model.state_dict(), model.pretrained_cfg


def conv_filter(state_dict, patch_size: int = 16):
    """
    Convert patch embedding weight from manual patchify + linear proj to conv
    """

    out_dict = {}
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict
