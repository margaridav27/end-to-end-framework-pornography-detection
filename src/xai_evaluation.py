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
DATA_LOC = "/nas-ctm01/datasets/public/BIOMETRICS/pornography-2k-db/data-processed/even-20"
RESULTS_LOC = "results/pornography-2k/cnns/data-aug/even-20"
STATE_DICT_LOC = os.path.join(
    RESULTS_LOC,
    "models",
    "vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20.pth",
)
EXPLANATIONS_LOC = os.path.join(
    RESULTS_LOC,
    "explanations",
    "vgg19_freeze_False_epochs_50_batch_16_optim_sgd_aug_True_split_10_20",
    "captum",
    "correct",
    "IG",
)

INPUT_SHAPE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32


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
    norm_std=NORM_STD
)["test"]
dataset = PornographyFrameDataset(
    data_loc=DATA_LOC, 
    df=load_split(DATA_LOC, split, "test")["test"], 
    transform=data_transforms
)
dataloader = DataLoader(
    dataset=dataset, 
    batch_size=BATCH_SIZE, 
    num_workers=8, 
    pin_memory=True
)


# To filter out the incorrect predictions
def filter_correct_predictions(preds, labels):
    return preds == labels


explain_func = generate_captum_explanations
explain_func_kwargs = {"method_name": "IG"}
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
                    EXPLANATIONS_LOC, "npys", f"{os.path.splitext(name)[0]}.npy"
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
            explain_func_kwargs=explain_func_kwargs,
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
with open(os.path.join(EXPLANATIONS_LOC, "quantus_evaluation.json"), "w") as f:
    json.dump(results, f, indent=2)

# Save average results to json by metric
with open(os.path.join(EXPLANATIONS_LOC, "quantus_evaluation_average.json"), "w") as f:
    json.dump(final_results, f, indent=2)
