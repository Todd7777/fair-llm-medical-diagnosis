#

import torch
import torch.nn as nn
import torchvision.models as models
import cnn_dataloaders
from tqdm import tqdm
import os
import yaml


# potentially use argparse to make optional arguments to pick exactly where to save model weights and whatever else
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


config = load_config("cnn_configs.yaml")


# using adam as optimizing alg
# num workers should = num cpu threads(for data loading), currently at 4 workers, batches of 64
class train_cnn:
    def __init__(
        self,
        model_name,
    ):
        self.name = config[model_name]["name"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = config[model_name]["lr"]
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader, self.eval_loader = cnn_dataloaders.make_cnn_dataloaders(
            config[model_name]["data_dir"],
            config[model_name]["metadata_path"],
            config[model_name]["batch_size"],
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

        print(model.classifier)
        print(model.classifier[1])
        print(type(model.classifier[1]))

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
        os.makedirs("cnn_weights", exist_ok=True)
        path = os.path.join("cnn_weights", f"{self.name}_fine_tuned.pt")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def train(self, num_epochs=10):
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
            ):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)  # forward pass
                loss = self.criterion(outputs, labels)
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
            for inputs, labels in self.eval_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")
        self.model.train()


def run_training(cnn_name):
    new_train = train_cnn(cnn_name)
    new_train.train()
    new_train.save_model()


cnn_name = "efficientnet"
run_training(cnn_name)
