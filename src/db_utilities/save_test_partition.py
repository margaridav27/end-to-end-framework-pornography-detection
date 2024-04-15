from src.utils.data import load_split
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import os
import argparse
import cv2


def _parse_arguments():
  parser = argparse.ArgumentParser(description="Setup APD-VIDEO dataset")
  parser.add_argument("--data_loc", type=str, required=True)
  parser.add_argument("--save_loc", type=str, required=True)
  parser.add_argument("--split", type=float, nargs="*", default=[0.1, 0.2], help="Validation and test")

  args = parser.parse_args()

  if not os.path.exists(args.data_loc):
    parser.error("Invalid --data_loc.")

  return args


def main():
  args = _parse_arguments()

  df_test = load_split(args.data_loc, args.split, ["test"])["test"]
  dataset = PornographyFrameDataset(args.data_loc, df_test)

  os.makedirs(args.save_loc, exist_ok=True)

  for i in range(len(dataset)):
    frame_name, frame, _ = dataset[i]
    cv2.imwrite(os.path.join(args.save_loc, frame_name), frame)


if __name__ == "__main__":
    main()
