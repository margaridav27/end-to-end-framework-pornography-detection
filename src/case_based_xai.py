from src.utils.misc import seed, set_device
from src.utils.data import load_split, get_transforms
from src.utils.model import parse_model_filename, load_model
from src.datasets.pornography_frame_dataset import PornographyFrameDataset

import os
import argparse
from typing import List
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


cases_filenames = [
  "vNonPorn000024#0.jpg",
  "vNonPorn000030#0.jpg",
  "vNonPorn000031#18.jpg",
  "vNonPorn000066#14.jpg",
  "vNonPorn000197#11.jpg",
  "vNonPorn000211#1.jpg",
  "vNonPorn000212#2.jpg",
  "vNonPorn000221#9.jpg",
  "vNonPorn000240#6.jpg",
  "vNonPorn000255#0.jpg",
  "vNonPorn000301#10.jpg",
  "vNonPorn000383#0.jpg",
  "vNonPorn000465#5.jpg",
  "vNonPorn000612#0.jpg",
  "vNonPorn000756#16.jpg",
  "vNonPorn000772#16.jpg",
  "vNonPorn000809#5.jpg",
  "vNonPorn000931#4.jpg",
  "vNonPorn000939#2.jpg",
  "vNonPorn000988#7.jpg",
  "vPorn000037#13.jpg",
  "vPorn000076#10.jpg",
  "vPorn000090#9.jpg",
  "vPorn000174#0.jpg",
  "vPorn000180#15.jpg",
  "vPorn000194#11.jpg",
  "vPorn000224#0.jpg",
  "vPorn000302#1.jpg",
  "vPorn000314#3.jpg",
  "vPorn000336#0.jpg",
  "vPorn000533#8.jpg",
  "vPorn000542#0.jpg",
  "vPorn000659#18.jpg",
  "vPorn000755#3.jpg",
  "vPorn000790#14.jpg",
  "vPorn000819#11.jpg",
  "vPorn000832#5.jpg",
  "vPorn000832#13.jpg",
  "vPorn000861#8.jpg",
  "vPorn000906#16.jpg",
]


def _parse_arguments():
  parser = argparse.ArgumentParser(description="Testing a trained pytorch model")
  parser.add_argument("--data_loc", type=str, required=True)
  parser.add_argument("--save_loc", type=str, required=True)
  parser.add_argument("--state_dict_loc", type=str, required=True)
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--input_shape", type=int, default=224)
  parser.add_argument("--norm_mean", type=float, nargs="*", default=[0.485, 0.456, 0.406])
  parser.add_argument("--norm_std", type=float, nargs="*", default=[0.229, 0.224, 0.225])
  
  args = parser.parse_args()

  if not os.path.exists(args.state_dict_loc):
      parser.error("Invalid --state_dict_loc argument.")

  return args


def _get_test_data(
  data_loc : str, 
  split : List[float], 
  batch_size : int,
  input_shape : int, 
  norm_mean : List[float], 
  norm_std : List[float]
) -> DataLoader:
  df_test = load_split(data_loc, split, ["test"])["test"]
  data_transforms = get_transforms(False, input_shape, norm_mean, norm_std)["test"]
  dataset = PornographyFrameDataset(data_loc, df_test, data_transforms)
  return dataset, DataLoader(dataset, batch_size)


def _get_feature_extractor(model : nn.Module):
  if isinstance(model, nn.Module):
    if hasattr(model.module, 'classifier'):
      if isinstance(model.module.classifier, nn.Sequential):
        return nn.Sequential(*(list(model.module.children())[:-1]))
      else:
        return nn.Sequential(*(list(model.module.children())[:-2]))
    elif hasattr(model.module, 'fc'):
      return nn.Sequential(*(list(model.module.children())[:-1]))
  else:
    raise ValueError("Model type not supported.")


def _extract_feature_vector(feature_extractor : nn.Module, input : torch.Tensor) -> torch.Tensor:
  feature_extractor.eval()

  with torch.no_grad():
    feature_vector = feature_extractor(input)

  return torch.flatten(feature_vector, start_dim=1)


def _calculate_cosine_similarity(batch : torch.Tensor, cases : List[torch.Tensor]) -> torch.Tensor:
  batch_norm = F.normalize(batch, p=2, dim=1)
  cases_norm = torch.stack([F.normalize(case, p=2, dim=1) for case in cases])
  similarities = torch.matmul(batch_norm, cases_norm.permute(1, 2, 0))
  return similarities.squeeze(0)


def _calculate_euclidean_distance(batch : torch.Tensor, cases : List[torch.Tensor]) -> torch.Tensor:
  distances = torch.cdist(batch, torch.stack(cases), p=2)
  return distances.squeeze(-1).T


def main():
  seed()

  args = _parse_arguments()

  device = set_device()

  model_filename, model_name, split = parse_model_filename(args.state_dict_loc)
  feature_extractor = _get_feature_extractor(load_model(model_name, args.state_dict_loc, device))
  dataset, dataloader = _get_test_data(
    args.data_loc,
    split,
    args.batch_size,
    args.input_shape,
    args.norm_mean,
    args.norm_std
  )

  # Extract the feature vector of each case frame
  cases_feature_vectors = []
  for filename in cases_filenames:
    _, case_input, _, _ = dataset[filename]
    case_input = case_input.to(device).unsqueeze(0)
    case_feature_vector = _extract_feature_vector(feature_extractor, case_input)
    cases_feature_vectors.append(case_feature_vector.to(device))

  results = {
    "frame": [],
    "cosine_case_1": [],
    "cosine_case_2": [],
    "cosine_case_3": [],
    "cosine_score_1": [],
    "cosine_score_2": [],
    "cosine_score_3": [],
    "euclidean_case_1": [],
    "euclidean_case_2": [],
    "euclidean_case_3": [],
    "euclidean_score_1": [],
    "euclidean_score_2": [],
    "euclidean_score_3": []
  }

  real_save_loc = os.path.join(args.save_loc, model_filename)
  os.makedirs(real_save_loc, exist_ok=True)

  K = 3
  for names, inputs, _, _ in dataloader:
    inputs = inputs.to(device)

    feature_vectors = _extract_feature_vector(feature_extractor, inputs)

    # Compute cosine similarities
    cosine_similarities_batch = _calculate_cosine_similarity(feature_vectors, cases_feature_vectors)  
    topk_cosine_batch_scores, topk_cosine_batch_indices = cosine_similarities_batch.topk(k=K, dim=1, largest=True)
    topk_cosine_batch_scores, topk_cosine_batch_indices = topk_cosine_batch_scores.T.cpu().numpy(), topk_cosine_batch_indices.T
    topk_cosine_batch_filenames = [[cases_filenames[i] for i in indices.tolist()] for indices in topk_cosine_batch_indices]

    # Compute euclidean distances
    euclidean_distances_batch = _calculate_euclidean_distance(feature_vectors, cases_feature_vectors)
    topk_euclidean_batch_scores, topk_euclidean_batch_indices = euclidean_distances_batch.topk(k=K, dim=1, largest=False)
    topk_euclidean_batch_scores, topk_euclidean_batch_indices = topk_euclidean_batch_scores.T.cpu().numpy(), topk_euclidean_batch_indices.T
    topk_euclidean_batch_filenames = [[cases_filenames[i] for i in indices.tolist()] for indices in topk_euclidean_batch_indices]

    results["frame"].extend(names)
    for i in range(K):
      results[f"cosine_case_{i+1}"].extend(topk_cosine_batch_filenames[i])
      results[f"cosine_score_{i+1}"].extend(topk_cosine_batch_scores[i])
      results[f"euclidean_case_{i+1}"].extend(topk_euclidean_batch_filenames[i])
      results[f"euclidean_score_{i+1}"].extend(topk_euclidean_batch_scores[i])

  pd.DataFrame(results).to_csv(f"{real_save_loc}/case_explanations.csv", index=False)


if __name__ == "__main__":
    main()
