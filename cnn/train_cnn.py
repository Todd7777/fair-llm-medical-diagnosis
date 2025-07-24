#

import torch
import torch.nn as nn
import torchvision.models as models
import cnn_dataloaders
from tqdm import tqdm
import os
import yaml
import argparse
from data.makedatasets.datasets import (
    RetinalImageDataset,
    ChestXRayDataset,
    PathologyImageDataset,
)


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
        "--num_epochs",
        type=int,
        required=True,
        help="Number of training epochs, validation happens every epoch",
    )
    parser.add_argument(
        "--dataset", required=True, help='"retinal", "pathology", "chestxray"'
    )
    return parser.parse_args()


args = parse_args()
config = load_config("cnn_configs.yaml")
seed = "NOT IMPLEMENTED"

DATASET_CLASSES = {
    "retinal": RetinalImageDataset,
    "pathology": PathologyImageDataset,
    "chestxray": ChestXRayDataset,
}


# using adam as optimizing alg
# num workers should = num cpu threads(for data loading), currently at 4 workers, batches of 64
class train_cnn:
    def __init__(
        self,
    ):
        self.name = args.model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = config[self.name]["training"]["lr"]
        self.criterion = nn.CrossEntropyLoss()  # If dataset is multiple diseases per image, use nn.BCEWithLogitsLoss instead of nn.CrossEntropyLoss
        self.train_loader = cnn_dataloaders.make_cnn_dataloader(
            data_args={
                "dataset_type": "train",
                "data_dir": args.data_dir,
                "metadata_dir": args.metadata_dir,
                "model_name": self.name,
            },
            dataset_class=DATASET_CLASSES[args.dataset],
            batch_size=config[self.name]["data"]["batch_size"],
        )
        self.eval_loader = cnn_dataloaders.make_cnn_dataloader(
            data_args={
                "dataset_type": "eval",
                "data_dir": args.data_dir,
                "metadata_dir": args.metadata_dir,
                "model_name": self.name,
            },
            dataset_class=DATASET_CLASSES[args.dataset],
            batch_size=config[self.name]["data"]["batch_size"],
        )

        num_classes = self.train_loader.dataset.get_num_classes()  # type: ignore as all the datasets have get_num_classes
        if self.name == "efficientnet":
            self.model = self._build_efficientnet(num_classes)
        elif self.name == "someothernet":
            self.model = self._build_someothernet(num_classes)

    def _build_efficientnet(self, num_classes):
        model = models.efficientnet_b0(weights="DEFAULT")
        # this efficientnet has 1280 features
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)  # type: ignore as it is a sequential, able to be indexed

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        return model.to(self.device)

    def _build_someothernet(self, num_classes):
        model = models.efficientnet_b0(weights="DEFAULT")  # find the right other cnn
        # this efficientnet has 1280 features
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)  # type: ignore as it is sequential, able to be indexed

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
        num_epochs = args.num_epochs
        self.model.train()

        os.makedirs("results", exist_ok=True)
        with open("results/train_results.txt", "w") as out_file:
            out_file.write(f"Training using seed: {seed}")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch in tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
            ):
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(inputs)  # forward pass
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                _, preds = torch.max(
                    outputs, 1
                )  # class with max probability for each sample in batch
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            acc = 100 * correct / total
            print(
                f"Epoch {epoch + 1}: Loss: {epoch_loss / len(self.train_loader):.4f} | Accuracy: {acc:.2f}%"
            )

            self.validate()

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.eval_loader:
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")
        with open("results/train_results.txt", "w") as out_file:
            out_file.write(f"Validation Accuracy: {acc:.2f}%")

        self.model.train()


def run_training():
    new_train = train_cnn()
    new_train.train()
    new_train.save_model()


run_training()
