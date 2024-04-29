from .vit import VisionTransformer

import timm


def _conv_filter(state_dict, patch_size=16):
    """convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


def vit_base_patch16_224(pretrained: bool = False, num_classes: int = 2):
    # https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
    timm_model = timm.create_model(
        model_name="vit_base_patch16_224.orig_in21k_ft_in1k",
        pretrained=pretrained,
        num_classes=num_classes
    )

    model = VisionTransformer(num_classes=num_classes, qkv_bias=True)
    model.default_cfg = timm_model.pretrained_cfg
    model.load_state_dict(_conv_filter(timm_model.state_dict()))

    return model


def vit_large_patch16_224(pretrained: bool = False, num_classes: int = 2):
    # https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    timm_model = timm.create_model(
        model_name="vit_large_patch16_224.orig_in21k",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    model = VisionTransformer(
        num_classes=num_classes, 
        embed_dim=1024, 
        depth=24, 
        num_heads=16, 
        qkv_bias=True
    )
    model.default_cfg = timm_model.pretrained_cfg
    model.load_state_dict(_conv_filter(timm_model.state_dict()))

    return model


def deit_base_patch16_224(pretrained: bool = False, num_classes: int = 2):
    # https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
    timm_model = timm.create_model(
        model_name="deit_base_patch16_224.fb_in1k",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    model = VisionTransformer(num_classes=num_classes, qkv_bias=True)
    model.default_cfg = timm_model.pretrained_cfg
    model.load_state_dict(timm_model.state_dict())

    return model
