""" Model creation / weight loading / state_dict helpers

Hacked together by / Copyright 2020 Ross Wightman
"""

import math

import torch.utils.model_zoo as model_zoo


def load_pretrained(
    model, cfg=None, num_classes=1000, in_channels=3, filter_fn=None, strict=True
):
    if cfg is None:
        cfg = getattr(model, "default_cfg")
    if cfg is None or "url" not in cfg or not cfg["url"]:
        print("Pretrained model URL is invalid, using random initialization")
        return

    state_dict = model_zoo.load_url(cfg["url"], progress=False, map_location="cpu")
    if "model" in state_dict:  # checkpoint
        state_dict = state_dict["model"]

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_channels == 1:
        print(f"Converting first conv (conv1_name) pretrained weights from 3 to 1 channel")

        conv1_name = cfg["first_conv"]
        conv1_weight = state_dict[conv1_name + ".weight"].float()  # Some weights are in torch.half, ensure it's float for sum on CPU
        conv1_type = conv1_weight.dtype

        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)

        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + ".weight"] = conv1_weight  
    elif in_channels != 3:
        conv1_name = cfg["first_conv"]
        conv1_weight = state_dict[conv1_name + ".weight"].float()
        conv1_type = conv1_weight.dtype

        O, I, J, K = conv1_weight.shape
        if I != 3:
            print(f"Deleting first conv (conv1_name) from pretrained weights")

            del state_dict[conv1_name + ".weight"]
            strict = False
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations 
            # of the original RGB input layer weights that'd work better for specific cases
            print(f"Repeating first conv (conv1_name) weights in channel dim")
            
            repeat = int(math.ceil(in_channels / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
            conv1_weight *= 3 / float(in_channels)
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + ".weight"] = conv1_weight

    classifier_name = cfg["classifier"]
    if num_classes == 1000 and cfg["num_classes"] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + ".weight"]
        state_dict[classifier_name + ".weight"] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + ".bias"]
        state_dict[classifier_name + ".bias"] = classifier_bias[1:]
    elif num_classes != cfg["num_classes"]:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + ".weight"]
        del state_dict[classifier_name + ".bias"]
        strict = False

    model.load_state_dict(state_dict, strict=strict)
