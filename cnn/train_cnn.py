#

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import cnn_dataset_maker
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

sys.path.remove("..")


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
    parser.add_argument("--total_gpus", required=False, help="The total number of GPUs")
    parser.add_argument(
        "--exclude_gpus",
        required=False,
        help='Comma-separated list of GPU device IDs to exclude from use. E.g., "0,2"',
    )
    parser.add_argument(
        "--include_gpus",
        required=False,
        help='Comma-separated list of GPU device IDs to include for use. E.g., "0,2"',
    )
    return parser.parse_args()


args = parse_args()

if (args.exclude_gpus is not None and args.total_gpus is None) or (
    args.include_gpus is not None and args.total_gpus is None
):
    raise Exception("If exclude_gpus or include_gpus is used, so must be total_gpus")

config = load_config("cnn_configs.yaml")
seed = "NOT IMPLEMENTED"

DATASET_CLASSES = {
    "retinal": RetinalImageDataset,
    "pathology": PathologyImageDataset,
    "chestxray": ChestXRayDataset,
}


# using adam as optimizing alg
# num workers should = num cpu threads(for data loading), currently at 4 workers, batches of 64
class TrainCnn:
    def __init__(self, device_ids):
        self.name = args.model_name

        if device_ids is None or len(device_ids) < 1:
            self.device_ids = [0]
        else:
            self.device_ids = device_ids

        if (
            self.device_ids is not None
            and len(self.device_ids) > 0
            and torch.cuda.is_available()
        ):
            self.device = torch.device(f"cuda:{self.device_ids[0]}")
        else:
            self.device = torch.device("cpu")

        print("Using main device:", self.device)
        self.num_workers = args.num_workers
        train_dataset = cnn_dataset_maker.make_cnn_dataset(
            data_args={
                "dataset_type": "train",
                "data_dir": args.data_dir,
                "metadata_dir": args.metadata_dir,
                "model_name": self.name,
            },
            dataset_class=DATASET_CLASSES[args.dataset],
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=len(self.device_ids) * config[self.name]["data"]["batch_size"],
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        eval_dataset = cnn_dataset_maker.make_cnn_dataset(
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
            batch_size=len(self.device_ids) * config[self.name]["data"]["batch_size"],
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

        self.lr = config[self.name]["training"]["lr"]
        self.criterion = nn.CrossEntropyLoss()  # If dataset is multiple diseases per image, use nn.BCEWithLogitsLoss instead of nn.CrossEntropyLoss
        self.weight_decay = config[self.name]["training"]["weight_decay"]
        self.optimizer = None
        self.warmup_scheduler = None

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

        if self.device_ids is not None and len(self.device_ids) > 1:
            print(f"Using {len(self.device_ids)} devices")
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
            self.model = self.model.cuda(device_ids[0])
        else:
            # single GPU or CPU
            self.model = self.model.to(self.device)

        if "warmup_steps" in config[self.name]["training"]:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,  # type: ignore as will always be instantiated
                start_factor=0.1,
                total_iters=config[self.name]["training"]["warmup_steps"],
            )

    # all pretrained on imagenet
    def _build_efficientnet_v2(self, num_classes):
        model = models.efficientnet_v2_m(  # s, m, l
            weights="DEFAULT"
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)  # type: ignore as it is a sequential, able to be indexed

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        return model.to(self.device)

    def _build_densenet(self, num_classes):
        model = models.densenet121(
            weights="DEFAULT"
        )  # may want to find a cnn not trained on imagenet
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
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

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
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
        out_file = open("results/train_results.txt", "w")
        out_file.write(f"Training using seed: {seed}\n")

        num_epochs = config[self.name]["training"]["epochs"]

        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch in tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
            ):
                inputs = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                outputs = self.model(inputs)  # forward pass
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()  # type: ignore as optimizer is instantiated
                loss.backward()
                self.optimizer.step()  # type: ignore
                if self.warmup_scheduler is not None:
                    self.warmup_scheduler.step()

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
                self.early_stopping.val_loss_best, model_to_pass
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
                correct += (preds == labels).sum().item()
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
    args = parse_args()

    total_gpus = int(args.total_gpus) if args.total_gpus else torch.cuda.device_count()
    available_gpus = list(range(total_gpus))

    if args.exclude_gpus:
        exclude = set(int(x) for x in args.exclude_gpus.split(","))
        available_gpus = [g for g in available_gpus if g not in exclude]

    if args.include_gpus:
        include = set(int(x) for x in args.include_gpus.split(","))
        available_gpus = [g for g in available_gpus if g in include]

    if available_gpus:
        device_ids = available_gpus
    else:
        device_ids = None
    print(f"Using GPUs: {device_ids}")

    trainer = TrainCnn(device_ids)
    trainer.train()


if __name__ == "__main__":
    main()
