"""
Metrics:

Faithfulness 
    - Faithfulness Correlation
    - Selectivity
    - Region Perturbation
Robustness 
    - Consistency
    - Max-Sensitivity
    - Relative Input Stability
    - Relative Output Stability
    - Relative Representation Stability
Complexity 
    - Sparseness
    - Complexity
"""

from src.utils.misc import set_device
from src.utils.data import load_split, get_transforms
from src.utils.model import parse_model_filename, load_model, predict
from src.utils.xai_captum import generate_captum_explanations
from src.utils.xai_zennit import generate_zennit_explanations
from src.utils.xai_evaluation import Selectivity
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import os
import json
from typing import List, Tuple
import numpy as np
import quantus

import torch
from torch.utils.data import DataLoader


# Constants
INPUT_SHAPE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 4

DATA_LOC = "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20"
RESULTS_LOC = "results/pornography-2k/cnns/data-aug/even-20"
STATE_DICT_LOC = os.path.join(RESULTS_LOC, "models", "vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth")

METHODS = {
    "captum": {
        "IG": {"method_name": "IG"},
        "DEEP-LIFT": {"method_name": "DEEP-LIFT"},
        "LRP": {"method_name": "LRP"},
        "LRP-CMP": {"method_name": "LRP-CMP"},
        "OCC": {
            "method_name": "OCC",
            "attribute_kwargs": {
                "sliding_window_shapes": (3, 8, 8),
                "strides": (3, 4, 4),
            },
        },
    },
    "zennit": {
        "IntegratedGradients": {
            "method_name": "IntegratedGradients",
            "method_kwargs": {"n_iter": 50},
        },
        "Gradient_EpsilonGammaBox": {
            "method_name": "Gradient",
            "composite_name": "EpsilonGammaBox",
            "composite_kwargs": {"low": -2.12, "high": 2.64},
        },
        "Gradient_EpsilonPlusFlat": {
            "method_name": "Gradient",
            "composite_name": "EpsilonPlusFlat",
        },
    },
}

METRICS = {
    # Faithfulness
    "faithfulness_corr": quantus.FaithfulnessCorrelation(
        nr_runs=100,
        subset_size=224,
        perturb_baseline="black",
        disable_warnings=True,
        return_aggregate=False,
    ),
    "selectivity": Selectivity(
        patch_size=8,
        perturb_baseline="black",
        disable_warnings=True,
        return_aggregate=False,
    ),
    "reg_perturbation": quantus.RegionPerturbation(
        patch_size=8,
        order="morf",  # most relevant first
        regions_evaluation=100,
        perturb_baseline="black",
        disable_warnings=True,
        return_aggregate=False,
    ),
    # Robustness
    "consistency": quantus.Consistency(
        disable_warnings=True, 
        return_aggregate=False
    ),
    "max_sensitivity": quantus.MaxSensitivity(
        nr_samples=10,
        return_nan_when_prediction_changes=False,
        disable_warnings=True,
        return_aggregate=False,
    ),
    "rel_inp_stability": quantus.RelativeInputStability(
        nr_samples=10,
        return_nan_when_prediction_changes=False,
        disable_warnings=True,
        return_aggregate=False,
    ),
    "rel_out_stability": quantus.RelativeOutputStability(
        nr_samples=10,
        return_nan_when_prediction_changes=False,
        disable_warnings=True,
        return_aggregate=False,
    ),
    "rel_rep_stability": quantus.RelativeRepresentationStability(
        nr_samples=10,
        return_nan_when_prediction_changes=False,
        disable_warnings=True,
        return_aggregate=False,
    ),
    # Complexity
    "sparseness": quantus.Sparseness(
        disable_warnings=True, 
        return_aggregate=False
    ),
    "complexity": quantus.Complexity(
        disable_warnings=True, 
        return_aggregate=False
    ),
}


def get_explanations_loc(library: str, method: str):
    """
    Get the explanations directory based on library and method.
    """

    return os.path.join(
        RESULTS_LOC,
        "explanations",
        "vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20",
        library,
        "correct",
        method,
    )


def get_correctly_classified(
    names: List[str], 
    inputs: torch.Tensor, 
    preds: torch.Tensor, 
    explanations_loc: str
):
    """
    Get explanations for the set of correctly classified input samples.
    Important: Assumes that explanations_loc only contains explanations for correctly classified input samples.
    """

    filtered_names = []
    filtered_inputs = []
    filtered_preds = []
    explanations = []

    for i, name in enumerate(names):
        np_file = os.path.join(explanations_loc, "npys", f"{os.path.splitext(name)[0]}.npy")

        if os.path.isfile(np_file):
            filtered_names.append(name)
            filtered_inputs.append(inputs[i])
            filtered_preds.append(preds[i])
            explanations.append(np.load(np_file))

    return (
        filtered_names,
        np.array(filtered_inputs),
        np.array(filtered_preds),
        np.array(explanations),
    )


# Set device
device = set_device()

# Load model
model_filename, model_name, split = parse_model_filename(STATE_DICT_LOC)
model = load_model(model_name, STATE_DICT_LOC, device)
model.eval()

# Load test data
data_transforms = get_transforms(
    data_aug=False,
    input_shape=INPUT_SHAPE,
    norm_mean=NORM_MEAN,
    norm_std=NORM_STD,
)["test"]
dataset = PornographyFrameDataset(
    data_loc=DATA_LOC,
    df=load_split(DATA_LOC, split, "test")["test"],
    transform=data_transforms,
)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    num_workers=8,
    pin_memory=True,
)


for library, methods in METHODS.items():
    explain_func = (
        generate_captum_explanations
        if library == "captum"
        else generate_zennit_explanations
    )

    for method, kwargs in methods.items():
        print(f"Evaluating {library}'s {method} explanations")

        explanations_loc = get_explanations_loc(library, method)

        # To ensure that channels dimension is equal to 1 on explain_func call
        kwargs["reduce_channels"] = True

        results = {}
        for names, inputs, _, _ in dataloader:
            inputs = inputs.to(device)
            _, preds = predict(model, inputs)

            filtered_names, filtered_inputs, filtered_preds, explanations = (
                get_correctly_classified(
                    names, inputs.cpu(), preds.cpu(), explanations_loc
                )
            )

            if len(filtered_names) == 0: continue

            # Metrics expect explanations to have channels dimension equal to 1
            if explanations.shape[1] != 1:
                explanations = np.sum(explanations, axis=1, keepdims=True)

            for key, metric in METRICS.items():
                scores = metric(
                    model=model,
                    x_batch=filtered_inputs,
                    y_batch=filtered_preds,
                    a_batch=explanations,
                    explain_func=explain_func,
                    explain_func_kwargs=kwargs,
                    softmax=False,
                    device=device,
                    batch_size=len(explanations),
                )

                if key == "selectivity":
                    scores = metric.get_auc_score

                if key in results:
                    results[key].update(dict(zip(filtered_names, scores)))
                else:
                    results[key] = dict(zip(filtered_names, scores))

        final_results = {}
        for metric_key, res_values in results.items():
            res_values_list = list(res_values.values())

            if metric_key in ["rrs", "max_sensitivity"]:
                final_results[metric_key] = np.max(res_values_list, axis=0)
            else:
                final_results[metric_key] = np.mean(res_values_list, axis=0)

            print(f"{metric_key}: {final_results[metric_key]}")

        # Save results to json by metric and image
        with open(os.path.join(explanations_loc, "quantus_evaluation.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Save average results to json by metric
        with open(os.path.join(explanations_loc, "quantus_evaluation_average.json"), "w") as f:
            json.dump(final_results, f, indent=2)
