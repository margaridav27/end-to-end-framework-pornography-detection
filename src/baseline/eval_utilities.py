from sklearn.metrics import (
  accuracy_score, 
  precision_score, 
  recall_score, 
  f1_score
)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def calculate_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return accuracy, precision, recall, f1


def calculate_iou(labels, preds):
    intersection = (preds & labels).float().sum((1, 2))
    union = (preds | labels).float().sum((1, 2))
    iou = (intersection / union).mean().item()
    return iou


def save_train_val_curves(save_loc, values, metric):
    plt.figure(figsize=(10, 5))
    plt.plot(values["train"], label=f'Train {metric}')
    plt.plot(values["val"], label=f'Validation {metric}')
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric}')
    plt.title(f'Training and Validation {metric} Curves')

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.legend()
    plt.savefig(save_loc)
