"""
Metrics:

Faithfulness 
    - Faithfulness Correlation
    - Selectivity
    - Region Perturbation
Robustness 
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
from src.utils.model import parse_model_filename
from src.utils.xai_evaluation import Selectivity, RegionPerturbation
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import src.interpretable_transformers.vit_config as ViTs
from src.interpretable_transformers.vit_config import *
from src.interpretable_transformers.xai_utils import generate_transformer_explanations

import os
import json
import numpy as np
import quantus

import torch
import torch.nn as nn


# Constants
DATA_LOC = "/nas-ctm01/datasets/public/BIOMETRICS/pornography-database/data-processed/even-20"
RESULTS_LOC = "results/pornography-800/transformers/no-freeze/even-20"
MODEL_NAME = "vit_base_patch16_224"
MODEL_NAME_FULL = MODEL_NAME + "_epochs_50_batch_16_optim_sgd_aug_False_split_10_20"
STATE_DICT_LOC = os.path.join(RESULTS_LOC, "models", MODEL_NAME_FULL + ".pth")
EXPLANATIONS_LOC = os.path.join(RESULTS_LOC, "explanations", MODEL_NAME_FULL, "correct")

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
    "reg_perturbation": RegionPerturbation(
        patch_size=8,
        order="morf",  # most relevant first
        regions_evaluation=100,
        perturb_baseline="black",
        disable_warnings=True,
        return_aggregate=False,
    ),
    # Robustness
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
    # Complexity
    "sparseness": quantus.Sparseness(disable_warnings=True, return_aggregate=False),
    "complexity": quantus.Complexity(disable_warnings=True, return_aggregate=False),
}


device = set_device()

model_filename, _, split = parse_model_filename(STATE_DICT_LOC)

constructor = getattr(ViTs, MODEL_NAME, None)
assert constructor is not None, "Invalid --model_name argument."

NUM_CLASSES = 2
model = constructor(num_classes=NUM_CLASSES)
model = nn.DataParallel(model)
model = model.to(device)

state_dict = torch.load(STATE_DICT_LOC, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Load test data
cfg = model.module.default_cfg
data_transforms = get_transforms(
    data_aug=False,
    input_shape=cfg["input_size"][1],
    norm_mean=cfg["mean"],
    norm_std=cfg["std"],
)["test"]
dataset = PornographyFrameDataset(
    data_loc=DATA_LOC,
    df=load_split(DATA_LOC, split, "test")["test"],
    transform=data_transforms,
)


results = {}
for name, input, label, _ in dataset:
    np_file = os.path.join(EXPLANATIONS_LOC, "npys", f"{os.path.splitext(name)[0]}.npy")

    # Skip if no explanation is available
    # Assumes that the explanations directory contains only explanations for correctly classified samples
    if not os.path.isfile(np_file):
        continue

    explanation = np.load(np_file) # NumPy array with shape (H,W)
    if len(explanation.shape) == 2:
        explanation = np.expand_dims(explanation, axis=0) # Convert to shape (1,H,W)

    # Metrics expect explanations to have channels dimension equal to 1
    if explanation.shape[0] != 1:
        explanation = np.sum(explanation, axis=0, keepdims=True)

    for key, metric in METRICS.items():
        scores = metric(
            model=model,
            x_batch=np.array([input]),
            y_batch=np.array([label]),
            a_batch=np.array([explanation]),
            explain_func=generate_transformer_explanations,
            explain_func_kwargs={"expand_dims": True},
            softmax=False,
            device=device,
        )

        if key == "selectivity" or key == "reg_perturbation":
            scores = metric.get_auc_score

        if key in results:
            results[key].update(dict(zip([name], scores)))
        else:
            results[key] = dict(zip([name], scores))


final_results = {}
for metric_key, res_values in results.items():
    res_values_list = list(res_values.values())

    if metric_key in ["rrs", "max_sensitivity"]:
        final_results[metric_key] = np.max(res_values_list, axis=0)
    else:
        final_results[metric_key] = np.mean(res_values_list, axis=0)

    print(f"{metric_key}: {final_results[metric_key]}")

# Save results to json by metric and image
with open(os.path.join(EXPLANATIONS_LOC, "quantus_evaluation.json"), "w") as f:
    json.dump(results, f, indent=2)

# Save average results to json by metric
with open(os.path.join(EXPLANATIONS_LOC, "quantus_evaluation_average.json"), "w") as f:
    json.dump(final_results, f, indent=2)
