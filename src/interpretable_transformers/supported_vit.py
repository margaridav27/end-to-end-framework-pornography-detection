from src.interpretable_transformers.helpers import load_timm_model, conv_filter
from src.interpretable_transformers.vit_lrp import VisionTransformer


def vit_base_patch16_224(pretrained=False, num_classes: int = 2):
    state_dict, cfg = load_timm_model(
        model_name="vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    state_dict = conv_filter(state_dict=state_dict)

    model = VisionTransformer(num_classes=num_classes, qkv_bias=True)
    model.load_state_dict(state_dict)

    return model, cfg
