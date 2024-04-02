import random
import numpy as np
import datetime
import torch


def format_time(elapsed):
  '''
    Takes a time in seconds and returns a string hh:mm:ss
  '''
  
  elapsed_rounded = int(round((elapsed))) # Round to the nearest second
  return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss


def seed():
  seed = 42
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)


def set_device():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Device:", device)
  return device
