from src.utils.data import load_split, get_transforms
from src.utils.model import parse_model_filename, load_model
from src.utils.explainability import generate_explanations, ATTRIBUTION_METHODS, NOISE_TUNNEL_TYPES
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import os
import ast
import argparse
from typing import List

import torch
import torch.nn as nn


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Training a pytorch model to classify pornographic content")
    parser.add_argument("--state_dict_loc", type=str, required=True)
    parser.add_argument("--data_loc", type=str, required=True)
    parser.add_argument("--save_loc", type=str, required=True)
    parser.add_argument("--filter", type=str, default="correct", help="Filter for predictions to generate explanations. Options: 'all' (all predictions), 'correct' (only correct predictions), 'incorrect' (only incorrect predictions). Default is 'correct'.")
    parser.add_argument("--partition", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=16, help="If --to_explain is passed, this will not be taken into consideration.")
    parser.add_argument("--input_shape", type=int, default=224)
    parser.add_argument("--norm_mean", type=float, nargs="*", default=[0.485, 0.456, 0.406])
    parser.add_argument("--norm_std", type=float, nargs="*", default=[0.229, 0.224, 0.225])
    parser.add_argument("--to_explain", type=str, nargs="*", default=[], help="Frame names for which an explanation is desired. If no names are given, an explanation for each prediction will be generated.")
    parser.add_argument("--method", type=str, required=True, help="Method to generate the explanation.")
    parser.add_argument("--method_kwargs", type=str, help="JSON string representing keyword arguments for initializing the attribution method.")
    parser.add_argument("--attribute_kwargs", type=str, help="JSON string representing keyword arguments for calling the attribute method.")
    parser.add_argument("--noise_tunnel", action="store_true", default=False)
    parser.add_argument("--noise_tunnel_type", type=str, default="SGSQ", help="NoiseTunnel smoothing type. Ignored if --noise_tunnel is False.")
    parser.add_argument("--noise_tunnel_samples", type=int, default=5, help="Number of randomly generated examples per sample. Ignored if --noise_tunnel is False.")
    
    args = parser.parse_args()

    if not os.path.exists(args.state_dict_loc):
        raise ValueError("Invalid --state_dict_loc argument.")

    if args.method not in ATTRIBUTION_METHODS.keys():
        parser.error("Invalid --method.")
        
    if args.noise_tunnel and args.noise_tunnel_type not in NOISE_TUNNEL_TYPES.keys():
        parser.error("Invalid --noise_tunnel_type.")

    if args.method_kwargs:
        try:
            args.method_kwargs = ast.literal_eval(args.method_kwargs)
        except (SyntaxError, ValueError):
            parser.error("Invalid --method_kwargs.")

    if args.attribute_kwargs:
        try:
            args.attribute_kwargs = ast.literal_eval(args.attribute_kwargs)
        except (SyntaxError, ValueError):
            parser.error("Invalid --attribute_kwargs.")

    return args


def _load_dataset(
    data_loc : str, 
    split : List[float], 
    partition : str,
    input_shape : int, 
    norm_mean : List[float], 
    norm_std : List[float]
) -> PornographyFrameDataset:
    if not os.path.exists(data_loc):
        raise ValueError("Invalid --data_loc argument.")

    print("Loading dataset...")
    df_test = load_split(data_loc, split, [partition])[partition]
    data_transforms = get_transforms(False, input_shape, norm_mean, norm_std)[partition]
    return PornographyFrameDataset(data_loc, df_test, data_transforms)


def main():
    args = _parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_filename, model_name, split = parse_model_filename(args.state_dict_loc)
    dataset = _load_dataset(args.data_loc, split, args.partition, args.input_shape, args.norm_mean, args.norm_std)
    model = load_model(model_name, args.state_dict_loc, device)
    model.eval()

    generate_explanations(
        save_loc=os.path.join(args.save_loc, model_filename, args.filter),
        model=model, 
        filter=args.filter,
        device=device,
        dataset=dataset,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std, 
        method_key=args.method, 
        method_kwargs=args.method_kwargs, 
        attribute_kwargs=args.attribute_kwargs,
        to_explain=args.to_explain, 
        batch_size=args.batch_size,
        noise_tunnel=args.noise_tunnel, 
        noise_tunnel_type=args.noise_tunnel_type, 
        noise_tunnel_samples=args.noise_tunnel_samples
    )


if __name__ == "__main__":
    main()
