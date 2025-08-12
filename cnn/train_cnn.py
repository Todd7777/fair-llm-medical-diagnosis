#

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import argparse
import sys

sys.path.append("..")
import training_utils.early_stopping as early_stopping
from data.makedatasets.datasets import (
    RetinalImageDataset,
    ChestXRayDataset,
    PathologyImageDataset,
)
from data.makedatasets import dataset_maker

sys.path.remove("..")

import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CUDA deterministic behavior (may reduce performance)

    # torch.use_deterministic_algorithms(True) (formerly torch.set_deterministic(True)) - more comprehensive, raises runtime error if operation has no deterministic implementation
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_seed(seed)


# potentially use argparse to make optional arguments to pick exactly where to save model weights and whatever else
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# the values that should be changed, yaml values shouldnt be an option under normal circumstances
def parse_args():
    parser = argparse.ArgumentParser(description="Test CNN with configurable paths")
    parser.add_argument(
        "--weights_dir", required=True, help="Directory containing model weights"
    )
    parser.add_argument(
        "--data_dir", required=True, help="Directory containing image data"
    )
    parser.add_argument(
        "--metadata_dir", required=True, help="DIRECTORY containing metadata files"
    )
    parser.add_argument("--model_name", required=True, help="Model name in yaml config")
    parser.add_argument(
        "--dataset", required=True, help='"retinal", "pathology", "chestxray"'
    )
    parser.add_argument(
        "--num_workers",
        required=True,
        type=int,
        help="Should be less than or equal to the number of cores",
    )
    parser.add_argument(
        "--gpu",
        required=False,
        help="Choose the gpu to use. Ex. 0",
    )
    return parser.parse_args()


args = parse_args()

if args.gpu is not None:
    torch.cuda.set_device(int(args.gpu))
elif torch.cuda.is_available():
    torch.cuda.set_device(0)

config = load_config("cnn_configs.yaml")

DATASET_CLASSES = {
    "retinal": RetinalImageDataset,
    "pathology": PathologyImageDataset,
    "chestxray": ChestXRayDataset,
}


# using adam as optimizing alg
# num workers should = num cpu threads(for data loading), currently at 4 workers, batches of 64
class TrainCnn:
    def __init__(self):
        self.name = args.model_name

        if args.gpu is not None:
            self.device = torch.device(f"cuda:{args.gpu}")
        else:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        self.num_workers = args.num_workers
        train_dataset = dataset_maker.make_cnn_dataset(
            data_args={
                "dataset_type": "train",
                "data_dir": args.data_dir,
                "metadata_dir": args.metadata_dir,
                "model_name": self.name,
            },
            dataset_class=DATASET_CLASSES[args.dataset],
        )

        self.batch_size = config[self.name]["data"]["batch_size"]
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        eval_dataset = dataset_maker.make_cnn_dataset(
            data_args={
                "dataset_type": "eval",
                "data_dir": args.data_dir,
                "metadata_dir": args.metadata_dir,
                "model_name": self.name,
            },
            dataset_class=DATASET_CLASSES[args.dataset],
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

        self.num_batches = len(self.train_loader)
        self.num_epochs = config[self.name]["training"]["epochs"]
        self.lr = config[self.name]["training"]["lr"]
        self.criterion = nn.CrossEntropyLoss()  # If dataset is multiple diseases per image, use nn.BCEWithLogitsLoss instead of nn.CrossEntropyLoss

        self.weight_decay = config[self.name]["training"].get("weight_decay", 0)
        self.warmup_steps = self.num_batches * config[self.name]["training"].get(
            "warmup_epochs", 0
        )

        self.optimizer = None
        self.warmup_scheduler = None
        self.cosine_scheduler = None

        os.makedirs(args.weights_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            args.weights_dir, f"{self.name}_{args.dataset}_fine_tuned_best.pt"
        )

        num_classes = self.train_loader.dataset.get_num_classes()  # type: ignore as all the datasets have get_num_classes
        if self.name == "efficientnet_v2":
            self.model = self._build_efficientnet_v2(num_classes)
            self.early_stopping = early_stopping.EarlyStopping(
                patience=4, path=checkpoint_path
            )
        elif self.name == "densenet":
            self.model = self._build_densenet(num_classes)
            self.early_stopping = early_stopping.EarlyStopping(
                patience=6, path=checkpoint_path
            )
        elif self.name == "convnext":
            self.model = self._build_convnext(num_classes)
            self.early_stopping = early_stopping.EarlyStopping(
                patience=6, path=checkpoint_path
            )
        else:
            raise Exception("wrong model name")

        if "warmup_epochs" in config[self.name]["training"]:
            self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,  # type: ignore as will always be instantiated
                self.lr_lambda,
            )
        if "cosine_annealing" in config[self.name]["training"]:
            self.cosine_scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,  # type: ignore
                    T_0=config[self.name]["training"]["cosine_annealing"]["T_0"],
                    eta_min=config[self.name]["training"]["cosine_annealing"][
                        "eta_min"
                    ],
                )
            )

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            # Compute a multiplier that starts at start_lr/base_lr (0.1) and goes up to 1.0
            return 0.1 + (1.0 - 0.1) * (current_step / self.warmup_steps)
        else:
            # After warmup, keep multiplier at 1.0 (base LR)
            return 1.0

    # all pretrained on imagenet
    def _build_efficientnet_v2(self, num_classes):
        model = models.efficientnet_v2_m(  # s, m, l
            weights="DEFAULT"
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)  # type: ignore as it is a sequential, able to be indexed

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return model.to(self.device)

    def _build_densenet(self, num_classes):
        model = models.densenet121(
            weights="DEFAULT"
        )  # may want to find a cnn not trained on imagenet
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return model.to(self.device)

    def _build_convnext(self, num_classes):
        model = models.convnext_tiny(
            weights="DEFAULT"
        )  # convnext v2 exists not in pytorch, different sizes of that up to "huge" ~660 mil
        # Replace the classifier head
        in_features = (
            model.classifier[2].in_features
        )  # ConvNeXt classifier has a sequential with layers; layer 2 is Linear
        model.classifier[2] = nn.Linear(in_features, num_classes)  # type: ignore as it is a sequential, able to be indexed

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return model.to(self.device)

    def save_model(self):
        os.makedirs(args.weights_dir, exist_ok=True)
        path = os.path.join(
            args.weights_dir, f"{self.name}_{args.dataset}_fine_tuned.pt"
        )
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def train(self):
        model_to_pass = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        os.makedirs("results", exist_ok=True)
        out_file = open(
            os.path.join("results", f"{self.name}_{args.dataset}_train_results.txt"),
            "w",
        )
        out_file.write(f"Training using seed: {seed}\n")

        self.model.train()
        warmup_step_counter = 0
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, batch in enumerate(
                tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            ):
                inputs = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                outputs = self.model(inputs)  # forward pass

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()  # type: ignore

                print(f"Loading batch: {batch_idx}")
                if (
                    self.warmup_scheduler is not None
                    and self.warmup_steps > warmup_step_counter
                ):
                    self.warmup_scheduler.step()
                    warmup_step_counter += 1
                elif self.cosine_scheduler is not None:
                    self.cosine_scheduler.step(
                        epoch + batch_idx / self.num_batches  # type: ignore
                    )  # only if fixed batch use self.num_batches

                current_lr = self.optimizer.param_groups[0]["lr"]  # type: ignore
                print(f"Learning rate after batch {batch_idx + 1}: {current_lr:.6f}")

                self.optimizer.zero_grad()  # type: ignore as optimizer is instantiated

                epoch_loss += loss.item()
                _, preds = torch.max(
                    outputs, 1
                )  # class with max probability for each sample in batch
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            acc = 100 * correct / total
            print(
                f"Epoch {epoch + 1}:\nTraining Loss: {epoch_loss / len(self.train_loader):.4f} | Training Accuracy: {acc:.2f}%"
            )
            out_file.write(
                f"Epoch {epoch + 1}:\nTraining Loss: {epoch_loss / len(self.train_loader):.4f} | Training Accuracy: {acc:.2f}%\n"
            )

            val_loss = self.validate(out_file)

            self.early_stopping(val_loss, model_to_pass)
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

            print(
                "\nCuda memory allocated (GB):", torch.cuda.memory_allocated() / 1024**3
            )
            print(
                "Cuda max memory reserved (GB):",
                torch.cuda.max_memory_reserved() / 1024**3,
                "\n",
            )

        out_file.close()
        if self.early_stopping.early_stop is False:
            self.early_stopping.save_checkpoint(
                self.early_stopping.val_loss_best, model_to_pass, final_save=True
            )

    def validate(self, out_file):
        self.model.eval()
        correct = 0
        total = 0
        valid_loss = 0

        with torch.no_grad():
            for batch in self.eval_loader:
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                valid_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().to(self.device)
                total += labels.size(0)

        acc = 100 * correct / total
        avg_loss = valid_loss / len(self.eval_loader)
        print(f"Validation Loss: {avg_loss:.4f} | Validation Accuracy: {acc:.2f}%\n\n")
        out_file.write(
            f"Validation Loss: {avg_loss:.4f} | Validation Accuracy: {acc:.2f}%\n\n"
        )

        self.model.train()
        return avg_loss


def main():
    try:
        trainer = TrainCnn()
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        torch.cuda.empty_cache()
        print("Cleanup done, exiting.")


if __name__ == "__main__":
    main()
