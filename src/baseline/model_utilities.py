from baseline.eval_utilities import calculate_metrics, calculate_iou

import time
import datetime
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# TODO move this function to somewhere else
def format_time(elapsed):
  '''
    Takes a time in seconds and returns a string hh:mm:ss
  '''
  
  elapsed_rounded = int(round((elapsed))) # Round to the nearest second
  return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss


class MixedPooling(nn.Module):
  def __init__(self, kernel_size, stride=None, padding=0):
    super(MixedPooling, self).__init__()
    self.avg_pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
    self.max_pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

  def forward(self, x):
    avg_out = self.avg_pool(x)
    max_out = self.max_pool(x)
    return torch.cat((avg_out, max_out), dim=1)
    

def add_mixed_pooling_and_batch_norm(module):
  if isinstance(module, nn.Conv2d):
    # Add BatchNorm2d after each Conv2d 
    return nn.Sequential(module, nn.BatchNorm2d(module.out_channels), nn.ReLU(inplace=True))
  elif isinstance(module, nn.MaxPool2d):
    # Replace MaxPool2d with MixedPooling
    return MixedPooling(kernel_size=module.kernel_size, stride=module.stride, padding=module.padding)
  elif isinstance(module, nn.Sequential):
    # Recursively go through Sequential modules
    return nn.Sequential(*(add_mixed_pooling_and_batch_norm(m) for m in module))
  else:
    return module
  

def optimize_model(model):
  # Apply changes recursively
  for name, module in model.named_children():
    setattr(model, name, add_mixed_pooling_and_batch_norm(module))
  return model


def get_pytorch_model(model_name : str, weights : str=None):
  if model_name == "resnet50": 
    return models.resnet50(weights=weights)
  elif model_name == "resnet101": 
    return models.resnet101(weights=weights)
  elif model_name == "resnet152": 
    return models.resnet152(weights=weights)
  elif model_name == "densenet121": 
    return models.densenet121(weights=weights)
  elif model_name == "densenet169": 
    return models.densenet169(weights=weights)
  elif model_name == "densenet201": 
    return models.densenet201(weights=weights)
  elif model_name == "alexnet":
    return models.alexnet(weights=weights)
  elif model_name == "vgg16": 
    return models.vgg16(weights=weights)
  elif model_name == "vgg19": 
    return models.vgg19(weights=weights)
  elif model_name == "mobilenetv2":
    return models.mobilenet_v2(weights=weights)
  else: 
    raise ValueError(f"Invalid model_name {model_name}")


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
  else:
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, 2)

  return model


def init_model(
    model_name : str, 
    weights : str=None, 
    freeze_layers : bool=True,
    optimized : bool=False
): 
  model = get_pytorch_model(model_name, weights)
  
  if freeze_layers:
    for params in model.parameters(): 
      params.requires_grad = False

  # Parameters of newly constructed modules have requires_grad=True by default
  model = modify_last_fcl(model, model_name)

  return optimize_model(model) if optimized else model


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
  
  best_model = model.state_dict()
  best_acc = 0.0

  losses = { "train": [], "val": [] }
  accuracies = { "train": [], "val": [] }

  for epoch_i in range(n_epochs):
    print('========== Start Epoch {:} / {:} =========='.format(epoch_i + 1, n_epochs))
    
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
      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        # Track history if only in train
        with torch.set_grad_enabled(phase == "train"):
          outputs = model(inputs)
          loss = criterion(outputs, F.one_hot(labels, num_classes=2).float())
          _, preds = torch.max(outputs, 1)

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
      epoch_acc = running_corrects.double() / dataset_sizes[phase]
      
      print("{} Loss: {:.4f} | Acc: {:.4f}".format("Training" if phase == "train" else "Validation", epoch_loss, epoch_acc))
      losses[phase].append(epoch_loss)
      accuracies[phase].append(epoch_acc.item())

      if phase == "val":
        if epoch_acc > best_acc:
          best_acc = epoch_acc
          best_model = model.state_dict()

    print("Epoch took {:}".format(format_time(time.time() - t0)))
    print('=========== End Epoch {:} / {:} ===========\n'.format(epoch_i + 1, n_epochs))

  return best_model, best_acc, accuracies, losses
  
  
def train_model(
    model, 
    dataloaders,
    dataset_sizes,
    optimizer, 
    n_epochs, 
    device
):
  '''
    General function to train a model
  '''

  # This loss combines a Sigmoid layer and the BCELoss in one single class
  criterion = nn.BCEWithLogitsLoss() 

  params = list(filter(lambda p: p.requires_grad, model.parameters()))
  lr, momentum = 0.001, 0.9
  if optimizer == "adam": optim = torch.optim.Adam(params, lr)
  else: optim = torch.optim.SGD(params, lr, momentum)

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=7, gamma=0.1)

  # Measure the total training time for the whole run
  total_t0 = time.time()

  best_model, best_acc, accuracies, losses = run_epochs(
    model, 
    dataloaders, 
    dataset_sizes, 
    criterion, 
    optim, 
    scheduler,
    n_epochs, 
    device
  )

  print("Training complete!")
  print("Total training took {:}".format(format_time(time.time() - total_t0)))
  print("Best Acc: {:.4f}\n".format(best_acc))
  
  # Load best model
  model.load_state_dict(best_model)
  
  return model, accuracies, losses


def test_model(model, dataloader, device, save_loc):
  '''
    General function to test a model
  '''
    
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
  accuracy, precision, recall, f1 = calculate_metrics(targets, predictions)
  iou = calculate_iou(targets, predictions)
  
  print("Testing complete!")
  print("Total testing took {:}".format(format_time(time.time() - t0)))
  print("Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f} | IOU Score: {:.4f}".format(accuracy, precision, recall, f1, iou))

  pd.DataFrame({ "Target": targets, "Prediction": predictions }).to_csv(save_loc, index=False)
