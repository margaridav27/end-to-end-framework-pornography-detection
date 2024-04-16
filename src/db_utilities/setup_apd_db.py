from src.utils.data import log_split, save_split

import os
import argparse
import shutil
from multiprocessing import Pool
import pandas as pd
from typing import List

from sklearn.model_selection import train_test_split


def _parse_arguments():
  parser = argparse.ArgumentParser(description="Setup APD-VIDEO dataset")
  parser.add_argument("--data_loc", type=str, nargs="+", required=True)
  parser.add_argument("--save_loc", type=str, required=True)
  parser.add_argument("--corrupted_paths_loc", type=str)
  parser.add_argument("--split", type=float, nargs="*", default=[0.1, 0.2], help="Validation and test")

  args = parser.parse_args()

  for loc in args.data_loc:
    if not os.path.exists(loc):
      parser.error(f"Invalid --data_loc: {loc}.")

  return args


def _remove_corrupted_files(corrupted_paths_file : str, save_loc : str):
  with open(corrupted_paths_file, 'r') as file:
    corrupted_paths = file.readlines()

  for path in corrupted_paths:
    path = path.strip()
    _, filename = os.path.split(path)
    if os.path.exists(os.path.join(save_loc, filename)): 
      os.remove(os.path.join(save_loc, filename))
      print(f"Removed corrupted file {filename}")


def _create_data_file(data_loc : str) -> pd.DataFrame:
  data = { "frame": [], "label": [] }

  for file in os.listdir(data_loc):
    if file.startswith("."): continue

    data["frame"].append(file)
    data["label"].append(0 if "nonPorn" in file else 1)

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


def _copytree(source_dir, destination_dir):
  try:
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
    print(f"Files copied from {source_dir} to {destination_dir}")
  except Exception as e:
    print(f"Error copying files from {source_dir} to {destination_dir}: {e}")


def _copy_files_to_destination(source_dirs : List[str], destination_dir : str):
  os.makedirs(destination_dir, exist_ok=True)
  with Pool(processes=len(source_dirs)) as pool:
    pool.starmap(_copytree, [(dir, destination_dir) for dir in source_dirs])


def main():
  args = _parse_arguments()

  _copy_files_to_destination(args.data_loc, args.save_loc)
  if args.corrupted_paths_loc: 
    _remove_corrupted_files(args.corrupted_paths_loc, args.save_loc)
  df_data = _create_data_file(args.save_loc)
  _create_split_file(df_data, args.split, args.save_loc)  


if __name__ == "__main__":
    main()