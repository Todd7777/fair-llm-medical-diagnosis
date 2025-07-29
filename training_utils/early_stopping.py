import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(self, patience, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_best = float("inf")
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score >= self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(
            f"Validation loss decreased ({self.val_loss_best:.6f} --> {val_loss:.6f}).  Saving model."
        )

        if isinstance(model, nn.Module):
            torch.save(model.state_dict(), self.path)
        elif True:  # Replace True with whatever way to determine model type and therefore saving method
            pass
        else:
            print("WARNING: Model type not recognized for saving. No checkpoint saved.")

        self.val_loss_best = val_loss
