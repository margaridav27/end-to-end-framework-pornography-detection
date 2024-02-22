import matplotlib.pyplot as plt
from sklearn.metrics import (
  accuracy_score, 
  precision_score, 
  recall_score, 
  f1_score
)


def calculate_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return accuracy, precision, recall, f1


def save_train_val_curves(save_loc, values):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,5))
    
    
    ax0.set_title("Loss")
    ax0.plot(values["train_loss"], label=f"Train")
    ax0.plot(values["val_loss"], label=f"Validation")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")

    ax1.set_title("Accuracy")
    ax1.plot(values["train_acc"], label=f"Train")
    ax1.plot(values["val_acc"], label=f"Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center")

    fig.savefig(save_loc)
