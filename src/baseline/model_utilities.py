import time
import datetime
import pandas as pd

import torch
import torch.nn as nn
from torchvision import models


# TODO move this function to somewhere else
def format_time(elapsed):
  '''
    Takes a time in seconds and returns a string hh:mm:ss
  '''
  
  elapsed_rounded = int(round((elapsed))) # Round to the nearest second
  return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss


def init_model(
    model_name : str, 
    n_classes : int,
    weights : str="IMAGENET1K_V1", 
    freeze_layers : bool=True 
): 
  # Load pytorch model
  if model_name == "resnet50":
    model = models.resnet50(weights=weights)
  elif model_name == "densenet121":
    model = models.densenet121(weights=weights)
  elif model_name == "vgg16":
    model = models.vgg16(weights=weights)
  else:
    raise ValueError(f"Invalid model_name {model_name}")
  
  if freeze_layers:
    for params in model.parameters(): 
      params.requires_grad = False

  # Parameters of newly constructed modules have requires_grad=True by default
  if "densenet" in model_name:
    n_features = model.classifier.in_features
    model.classifier = nn.Linear(n_features, n_classes)
  elif "vgg" in model_name or "alexnet" in model_name:
    n_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(n_features, n_classes)])
    model.classifier = nn.Sequential(*features)
  else:
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_classes)

  return model


def run_epochs(
    model, 
    dataloaders,
    dataset_sizes,
    criterion, 
    optimizer, 
    scheduler,
    n_epochs,
    device
): 
  '''
    General function to run n_epochs epochs
  '''
  for epoch_i in range(n_epochs):
    print('========== Start Epoch {:} / {:} =========='.format(epoch_i + 1, n_epochs))
    
    # Measure the training time per epoch
    t0 = time.time()

    best_model = model.state_dict()
    best_acc = 0.0

    # Each epoch has a training and validation phase
    for phase in ["train", "val"]:
      if phase == "train":
        print("Training...")
        model.train() # Set model to training mode
      else:
        print("Running Validation...")
        model.eval() # Set model to evaluate mode

      run_loss = 0.0
      run_corrects = 0

      # Iterate over data
      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        # Track history if only in train
        with torch.set_grad_enabled(phase == "train"):
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          _, preds = torch.max(outputs, 1)

          # If in training phase, backward + optimize
          if phase == "train":
            loss.backward()
            optimizer.step()

        # Statistics 
        run_loss += loss.item()
        run_corrects += torch.sum(preds == labels)

      if phase == "train":
        scheduler.step()

      epoch_loss = run_loss / dataset_sizes[phase]
      epoch_acc = run_corrects / dataset_sizes[phase]

      print("{} Loss: {:.4f} | Acc: {:.4f}".format("Training" if phase == "train" else "Validation", epoch_loss, epoch_acc))

      if phase == "val":
        if epoch_acc > best_acc:
          best_acc = epoch_acc
          best_model = model.state_dict()
        
    print("Epoch took {:}".format(format_time(time.time() - t0)))
    print('=========== End Epoch {:} / {:} ===========\n'.format(epoch_i + 1, n_epochs))

  return best_model, best_acc
  
  
def train_model(
    model, 
    dataloaders,
    dataset_sizes,
    criterion, 
    optimizer, 
    scheduler,
    n_epochs, 
    device
):
  '''
    General function to train a model
  '''

  # Measure the total training time for the whole run
  total_t0 = time.time()

  best_model, best_acc = model.state_dict(), 0.0
  best_model, best_acc = run_epochs(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, n_epochs, device)

  print("Training complete!")
  print("Total training took {:}".format(format_time(time.time() - total_t0)))
  print("Best Acc: {:.4f}\n".format(best_acc))
  
  # Load best model
  model.load_state_dict(best_model)
  
  return model


def test_model(model, dataloader, dataset_size, device, save_loc):
  # Measure the testing time
  t0 = time.time()

  targets = []
  predictions = []

  corrects = 0

  model.eval() # Set model to evaluate mode

  with torch.no_grad():
    for inputs, labels in dataloader:
      inputs = inputs.to(device)
      labels = labels.to(device)

      outputs = model(inputs) # Forward pass

      _, preds = torch.max(outputs, 1)
      corrects += torch.sum(preds == labels)
      
      predictions.extend(preds.cpu().numpy())
      targets.extend(labels.cpu().numpy())

  # Compute final accuracy
  acc = corrects / dataset_size
  
  print("Testing complete!")
  print("Total testing took {:}".format(format_time(time.time() - t0)))
  print("Acc: {:.4f}\n".format(acc))

  # Export predictions
  print("Exporting predictions and ground-truth...")
  pd.DataFrame({ "Prediction": predictions, "Target": targets }).to_csv(save_loc, index=False)
