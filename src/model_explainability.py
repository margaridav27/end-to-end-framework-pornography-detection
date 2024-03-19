from src.utils.data import load_split, get_transforms
from src.utils.model import init_model, predict
from src.utils.evaluation import save_train_val_curves
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import os
import argparse
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from captum.attr import visualization as viz, IntegratedGradients

parser = argparse.ArgumentParser(description="Training a pytorch model to classify pornographic content")
parser.add_argument("--state_dict_loc", type=str, required=True)
parser.add_argument("--data_loc", type=str, required=True)
parser.add_argument("--save_loc", type=str, required=True)
parser.add_argument("--explainer", type=str, required=True, help="Method to generate the explanation.")
parser.add_argument("--to_explain", type=str, nargs="*", default=[], help="Frame names for which an explanation is desired. If no names are given, an explanation for each prediction will be generated.")
parser.add_argument("--batch_size", type=int, default=16, help="If --to_explain is passed, this will not be taken into consideration.")
parser.add_argument("--input_shape", type=int, default=224)
parser.add_argument("--norm_mean", type=float, nargs="*", default=[0.485, 0.456, 0.406])
parser.add_argument("--norm_std", type=float, nargs="*", default=[0.229, 0.224, 0.225])
args = parser.parse_args()

_, model_filename = os.path.split(args.state_dict_loc) # Includes .pth
model_filename = model_filename.split(".")[0] # Does not include .pth
model_name = model_filename.split("_")[0]

split = model_filename.split("_")[-2:]
split = [float(i)/100 for i in split]

print(f"Loading dataset...")
if not os.path.exists(args.data_loc):
    raise ValueError("Invalid --data_loc argument.")

df_test = load_split(args.data_loc, split, ["test"])["test"]
data_transforms = get_transforms(False, args.input_shape, args.norm_mean, args.norm_std)["test"]
dataset = PornographyFrameDataset(args.data_loc, df_test, data_transforms)

print(f"Loading {model_name}...")
if not os.path.exists(args.state_dict_loc):
    raise ValueError("Invalid --state_dict_loc argument.")

if not os.path.exists(args.save_loc):
    os.makedirs(args.save_loc)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

state_dict = torch.load(args.state_dict_loc, map_location=device)
model = init_model(model_name)
model = torch.nn.DataParallel(model)
model.load_state_dict(state_dict)
# model = model.to(device)
model.eval()

method = IntegratedGradients(model)

attributions = {}
if args.to_explain:
    for frame_name in args.to_explain:
        name, input, label = dataset[frame_name]

        input = input.unsqueeze(0)
        input.requires_grad_()

        # input = input.to(device)
        # frame_label = frame_label.to(device)

        _, pred = predict(model, input)

        attr = method.attribute(inputs=input, target=pred, n_steps=10)
        attributions[name] = attr

        viz.visualize_image_attr(
            attr[0].cpu().permute(1,2,0).detach().numpy(),
            input.squeeze().cpu().permute(1,2,0).detach().numpy(),
            method="blended_heat_map",
            sign="positive",
            show_colorbar=True,
            title="Integrated Gradients",
            use_pyplot=False
        )
        plt.savefig(f"{args.save_loc}/{model_filename}_{name}.png")
else:
    dataloader = DataLoader(dataset, args.batch_size)
    for names, inputs, labels in dataloader:
        inputs.requires_grad_()

        # inputs = inputs.to(device)
        # labels = labels.to(device)

        _, pred = predict(model, inputs)

        attrs = method.attribute(inputs=inputs, target=labels, n_steps=10)
        for n, a in zip(names, attrs):
            attributions[n] = a
            viz.visualize_image_attr(
                a[0].cpu().permute(1,2,0).detach().numpy(),
                input.squeeze().cpu().permute(1,2,0).detach().numpy(),
                method="blended_heat_map",
                sign="positive",
                show_colorbar=True,
                title="Integrated Gradients",
                use_pyplot=False
            )
            plt.savefig(f"{args.save_loc}/{model_filename}_{n}.png")

