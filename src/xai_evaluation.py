"""
Metrics:

Faithfulness 
    - Selectivity
    - Faithfulness Correlation
Robustness 
    - Max-Sensitivity
    - Avg-Sensitivity
    - Relative Representation Stability (RRS)
Complexity 
    - Sparseness
    - Complexity
Localisation 
    - Attribution Localisation
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
import numpy as np
import quantus

from torch.utils.data import DataLoader


# Constants
INPUT_SHAPE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 8

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

DATA_LOC = "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20"
RESULTS_LOC = "results/pornography-2k/cnns/data-aug/even-20"
STATE_DICT_LOC = os.path.join(RESULTS_LOC, "models", "vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth")


# To get the explanations directory based on library and method
def get_explanations_loc(library, method):
    return os.path.join(
        RESULTS_LOC,
        "explanations",
        "vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20",
        library,
        "correct",
        method,
    )


# To filter out the incorrect predictions
def filter_correct_predictions(preds, labels):
    return preds == labels


# Set device
device = set_device()

# Load model
model_filename, model_name, split = parse_model_filename(STATE_DICT_LOC)
model = load_model(model_name, STATE_DICT_LOC, device)
model.eval()

# Load test data
data_transforms = get_transforms(
    data_aug=False, input_shape=INPUT_SHAPE, norm_mean=NORM_MEAN, norm_std=NORM_STD
)["test"]
dataset = PornographyFrameDataset(
    data_loc=DATA_LOC,
    df=load_split(DATA_LOC, split, "test")["test"],
    transform=data_transforms,
)
dataloader = DataLoader(
    dataset=dataset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True
)


metrics = {
    "selectivity": Selectivity(
        perturb_baseline="mean",
        patch_size=8,
        disable_warnings=True,
        return_aggregate=False,
    ),
    "faithfulness_corr": quantus.FaithfulnessCorrelation(
        perturb_baseline="mean",
        subset_size=224,
        nr_runs=100,
        disable_warnings=True,
        return_aggregate=False,
    ),
    "max_sensitivity": quantus.MaxSensitivity(
        nr_samples=8,
        return_nan_when_prediction_changes=False,
        disable_warnings=True,
        return_aggregate=False,
    ),
    "rrs": quantus.RelativeRepresentationStability(
        nr_samples=8,
        return_nan_when_prediction_changes=False,
        disable_warnings=True,
        return_aggregate=False,
    ),
    "sparseness": quantus.Sparseness(
        disable_warnings=True,
        return_aggregate=False
    ),
    "complexity": quantus.Complexity(
        disable_warnings=True,
        return_aggregate=False
    ),
    "attr_localisation": quantus.AttributionLocalisation(
        disable_warnings=True,
        return_aggregate=False
    ),
}


for library, methods in METHODS.items():
    explain_func = generate_captum_explanations if library == "captum" else generate_zennit_explanations
    
    for method, kwargs in methods.items():
        explanations_loc = get_explanations_loc(library, method)
        
        results = {}
        for names, inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            _, preds = predict(model, inputs)

            # Evaluate only on the set of correctly predicted input samples (skip if no correct predictions)
            mask = filter_correct_predictions(preds, labels)
            if mask.sum() == 0:
                continue

            names = [name for name, m in zip(names, mask) if m]
            inputs, preds = inputs[mask], preds[mask]

            # Get corresponding explanations
            explanations = np.array(
                [
                    np.load(
                        os.path.join(
                            explanations_loc, "npys", f"{os.path.splitext(name)[0]}.npy"
                        )
                    )
                    for name in names
                ]
            )

            # Check if there is an explanation for each input
            assert inputs.shape == explanations.shape, "Inputs shape must match explanations shape"

            for key, metric in metrics.items():
                scores = metric(
                    model=model,
                    x_batch=inputs.cpu().numpy(),
                    y_batch=preds.cpu().numpy(),
                    a_batch=explanations,
                    explain_func=explain_func,
                    explain_func_kwargs=kwargs,
                    device=device,
                    batch_size=len(explanations),
                )

                if key == "selectivity":
                    scores = metric.get_auc_score

                if key in results:
                    results[key].update(dict(zip(names, scores)))
                else:
                    results[key] = dict(zip(names, scores))

                print(results)

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
