from src.utils.misc import format_time
from src.utils.evaluation import calculate_metrics

import os
import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import wandb


def parse_model_filename(model_filename : str):
  _, model_filename = os.path.split(model_filename)
  model_filename_no_ext, _ = os.path.splitext(model_filename) # Does not include .pth

  model_filename_split = model_filename_no_ext.split("_")
  model_name = model_filename_split[0] if model_filename_split[1] == "freeze" else "_".join(model_filename_split[:2])

  split = [float(i)/100 for i in model_filename_no_ext.split("_")[-2:]]

  return model_filename_no_ext, model_name, split


def get_pytorch_model(model_name: str, weights: str = None):
  model_constructor = getattr(models, model_name, None)

  if model_constructor is None:
    raise ValueError(f"Invalid model_name {model_name}")

  try:
    return model_constructor(weights=weights)
  except:
    if weights == "IMAGENET1K_V1":
        return model_constructor(pretrained=True)
  

def modify_last_fcl(model, model_name):
  if model_name.startswith("densenet"):
    n_features = model.classifier.in_features
    model.classifier = nn.Linear(n_features, 2)
  elif model_name.startswith("vgg") or model_name.startswith("alexnet"):
    n_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(n_features, 2)])
    model.classifier = nn.Sequential(*features)
  elif model_name.startswith("mobilenet"):
    n_features = model.classifier[1].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(n_features, 2)])
    model.classifier = nn.Sequential(*features)
  else: # e.g., resnet
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, 2)

  return model


def init_model(
    model_name : str, 
    weights : str=None, 
    freeze_layers : bool=False
): 
  model = get_pytorch_model(model_name, weights)
  
  if freeze_layers:
    for params in model.parameters(): 
      params.requires_grad = False

  # Parameters of newly constructed modules have requires_grad=True by default
  model = modify_last_fcl(model, model_name)

  return model


def load_model(
    model_name : str, 
    state_dict_loc : str, 
    device : str,
) -> nn.Module :
    print(f"Loading {model_name}...")

    state_dict = torch.load(state_dict_loc, map_location=device)
    model = init_model(model_name)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model = model.to(device)

    return model


def train_model(
    model, 
    dataloaders,
    dataset_sizes,
    optim, 
    learning_rate,
    n_epochs, 
    device,
    wandb_on
):
  '''
    General function to train a model
  '''

  # This loss combines a Sigmoid layer and the BCELoss in one single class
  criterion = nn.BCEWithLogitsLoss() 

  params = list(filter(lambda p: p.requires_grad, model.parameters()))
  if optim == "adam": optimizer = torch.optim.Adam(params=params, lr=learning_rate)
  else: optimizer = torch.optim.SGD(params=params, lr=learning_rate, momentum=0.9)

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=0.1)

  # Measure the total training time for the whole run
  total_t0 = time.time()

  best_model = model.state_dict()
  best_acc = 0.0
  best_epoch = 1

  metrics = { "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [] }

  if wandb_on: wandb.watch(model, criterion=criterion, log="all", log_freq=10)

  for epoch_i in range(n_epochs):
    print('========== Start Epoch {} / {} =========='.format(epoch_i + 1, n_epochs))
    
    # Measure the training time per epoch
    t0 = time.time()
    
    # Each epoch has a training and validation phase
    for phase in ["train", "val"]:
      if phase == "train":
        print("Training...")
        model.train() # Set model to training mode
      else:
        print("Running Validation...")
        model.eval() # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data
      for _, inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        # Track history if only in train
        with torch.set_grad_enabled(phase == "train"):
          outputs = model(inputs)
          loss = criterion(outputs, F.one_hot(labels, num_classes=2).float())
          _, preds = torch.max(outputs, dim=1)

          # If in training phase, backward + optimize
          if phase == "train":
            loss.backward()
            optimizer.step()

        # Statistics 
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

      if phase == "train":
        scheduler.step()

      epoch_loss = running_loss / dataset_sizes[phase]
      metrics[f"{phase}_loss"].append(epoch_loss)

      epoch_acc = running_corrects.double() / dataset_sizes[phase]
      metrics[f"{phase}_acc"].append(epoch_acc.item())
      
      phase_name = "Training" if phase == "train" else "Validation"
      print("{} Loss: {:.4f} | Acc: {:.4f}".format(phase_name, epoch_loss, epoch_acc))
      if wandb_on:
        wandb.log({ 
          f"{phase_name} Loss": epoch_loss, 
          f"{phase_name} Accuracy": epoch_acc 
        }, step=epoch_i)
      
      if phase == "val" and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_epoch = epoch_i + 1
        best_model = model.state_dict()
        print("Updated best model")

    print("Epoch took {}".format(format_time(time.time() - t0)))
    print('=========== End Epoch {} / {} ===========\n'.format(epoch_i + 1, n_epochs))

  print("Training complete!")
  print("Total training took {}".format(format_time(time.time() - total_t0)))
  print("Best Acc: {:.4f} (epoch {})\n".format(best_acc, best_epoch))

  if wandb_on: wandb.finish()

  # Clear gradients
  optimizer.zero_grad()

  # Clear model, optimizer, criterion, and scheduler
  del model
  del optimizer
  del criterion
  del scheduler
  
  # Run garbage collector
  gc.collect()

  return best_model, metrics


def predict(model, input):
  # Forward pass
  outputs = model(input)

  # Applying sigmoid as it is a binary classification problem
  confidences, preds = torch.max(torch.sigmoid(outputs), dim=1) 

  return confidences, preds


def test_model(model, dataloader, device):
  '''
    General function to test a model
  '''
    
  # Measure the testing time
  t0 = time.time()

  frame_names = []
  targets = []
  predictions = []
  predictions_confidence = []

  model.eval() # Set model to evaluate mode

  with torch.no_grad():
    for names, inputs, labels in dataloader:
      inputs = inputs.to(device)
      labels = labels.to(device)

      confidences, preds = predict(model, inputs)

      frame_names.extend(names)
      targets.extend(labels.cpu().numpy())
      predictions.extend(preds.cpu().numpy())
      predictions_confidence.extend(confidences.cpu().numpy())

  # Compute final accuracy
  accuracy, precision, recall, f1 = calculate_metrics(targets, predictions)
  
  print("Testing complete!")
  print("Total testing took {:}".format(format_time(time.time() - t0)))
  print("Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f}".format(accuracy, precision, recall, f1))

  return { 
    "Frame": frame_names,
    "Target": targets, 
    "Prediction": predictions,
    "Confidence": predictions_confidence 
  }
