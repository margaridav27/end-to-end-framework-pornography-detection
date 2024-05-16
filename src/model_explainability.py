from src.utils.misc import set_device
from src.utils.data import load_split, get_transforms
from src.utils.model import parse_model_filename, load_model
from src.utils.explainability import generate_explanations, ATTRIBUTION_METHODS, NOISE_TUNNEL_TYPES
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import os
import gc
import ast
import argparse


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Generating explanations for a model's predictions using Captum library")
    parser.add_argument("--data_loc", type=str, required=True)
    parser.add_argument("--save_loc", type=str, required=True)
    parser.add_argument("--state_dict_loc", type=str, required=True)
    parser.add_argument("--filter", type=str, default="correct", choices=["all", "correct", "incorrect"], help="Filter for predictions to generate explanations. Options: 'all' (all predictions), 'correct' (only correct predictions), 'incorrect' (only incorrect predictions). Default is 'correct'.")
    parser.add_argument("--partition", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=4, help="If --to_explain is passed, this will not be taken into consideration.")
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
    parser.add_argument("--side_by_side", action="store_true", default=False)
    parser.add_argument("--show_colorbar", action="store_true", default=False)
    parser.add_argument("--colormap", type=str, default="jet")
    parser.add_argument("--outlier_perc", default=2)
    parser.add_argument("--alpha_overlay", type=float, default=0.5)

    args = parser.parse_args()

    if not os.path.exists(args.data_loc):
        raise ValueError("Invalid --data_loc argument.")

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


def main():
    args = _parse_arguments()

    device = set_device()

    model_filename, model_name, split = parse_model_filename(args.state_dict_loc)

    data_transforms = get_transforms(False, args.input_shape, args.norm_mean, args.norm_std)[args.partition]
    dataset = PornographyFrameDataset(
        data_loc=args.data_loc, 
        df=load_split(args.data_loc, split, args.partition)[args.partition], 
        transform=data_transforms
    )

    model = load_model(model_name, args.state_dict_loc, device)
    model.eval()

    generate_explanations(
        save_loc=os.path.join(args.save_loc, model_filename),
        model=model,
        filter=args.filter,
        device=device,
        dataset=dataset,
        method_key=args.method,
        method_kwargs=args.method_kwargs,
        attribute_kwargs=args.attribute_kwargs,
        noise_tunnel=args.noise_tunnel,
        noise_tunnel_type=args.noise_tunnel_type,
        noise_tunnel_samples=args.noise_tunnel_samples,
        to_explain=args.to_explain,
        batch_size=args.batch_size,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
        side_by_side=args.side_by_side,
        show_colorbar=args.show_colorbar,
        colormap=args.colormap,
        outlier_perc=args.outlier_perc,
        alpha_overlay=args.alpha_overlay,
    )

    # Clear model
    del model

    # Run garbage collector
    gc.collect()


if __name__ == "__main__":
    main()
