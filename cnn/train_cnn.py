#

import torch
import torch.nn as nn
import torchvision.models as models
import cnn_dataloaders
from tqdm import tqdm


# using adam as optimizing alg
# num workers should = num cpu threads(for data loading), currently at 4 workers, batches of 64
class train_cnn:
    def __init__(
        self,
        name,
        data_dir,
        metadata_path,
        num_classes,
        batch_size=64,
        lr=0.001,
    ):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        if name == "efficientnet":
            self.model = self._build_efficientnet(num_classes)
        elif name == "someothernet":
            self.model = self._build_someothernet(num_classes)

        self.train_loader, self.eval_loader = cnn_dataloaders.make_cnn_dataloaders(
            data_dir, metadata_path, batch_size
        )

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
        model = models.efficientnet_b0(weights="DEFAULT")
        # this efficientnet has 1280 features
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)  # type: ignore as it is sequential, able to be indexed

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        return model.to(self.device)

    def save_model(self):
        path = f"{self.name}.pth"  # TODO: join directory to it
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
