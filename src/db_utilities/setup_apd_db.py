from src.utils.data import log_split, save_split

import os
import argparse
import pandas as pd
from typing import List

from sklearn.model_selection import train_test_split


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Setup APD-VIDEO dataset")
    parser.add_argument("--data_loc", type=str, required=True)
    parser.add_argument("--split", type=float, nargs="*", default=[0.1, 0.2], help="Validation and test")

    args = parser.parse_args()

    if not os.path.exists(args.data_loc):
      parser.error("Invalid --data_loc.")

    return args


def _create_data_file(data_loc : str) -> pd.DataFrame:
  data = { "frame": [], "label": [] }

  for file in os.listdir(data_loc):
    if file.startswith("."): continue

    data["frame"] = file
    data["label"] = 0 if "nonPorn" in file else 1

  df_data = pd.DataFrame(data)
  df_data.to_csv(f"{data_loc}/data.csv", index=False)
  return df_data


def _create_split_file(df : pd.DataFrame, split_sizes : List[float], save_loc : str):
  val_size, test_size = split_sizes
  real_val_size = (1 - test_size) * val_size

  train, test = train_test_split(df, test_size=test_size, random_state=42)
  train, val = train_test_split(train, test_size=real_val_size, random_state=42)

  split = { "train": train, "val": val, "test": test }
  print("Created split for APD-VIDEO dataset.")
  log_split(split)
  save_split(save_loc, split_sizes, list(split.keys()), split)


def main():
  args = _parse_arguments()

  df_data = _create_data_file(args.data_loc)

  _create_split_file(df_data, args.split, args.data_loc)  


if __name__ == "__main__":
    main()