# using fine tuned model weights

import torch
import torchvision.models as models
import torch.nn as nn
import cnn_dataloaders
import yaml
from tqdm import tqdm
import argparse
import os
from data.makedatasets.datasets import (
    RetinalImageDataset,
    ChestXRayDataset,
    PathologyImageDataset,
)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# the values that should be changed, yaml values shouldnt be an option under normal circumstances
def parse_args():
    parser = argparse.ArgumentParser(description="Test CNN with configurable paths")
    parser.add_argument(
        "--weights_dir", required=False, help="Directory containing model weights"
    )
    parser.add_argument(
        "--data_dir", required=True, help="Directory containing image data"
    )
    parser.add_argument(
        "--metadata_dir", required=True, help="DIRECTORY containing metadata files"
    )
    parser.add_argument("--model_name", required=True, help="Model name in config")
    parser.add_argument(
        "--dataset", required=True, help='"retinal", "pathology", "chestxray"'
    )
    return parser.parse_args()


DATASET_CLASSES = {
    "retinal": RetinalImageDataset,
    "pathology": PathologyImageDataset,
    "chestxray": ChestXRayDataset,
}


args = parse_args()
config = load_config("cnn_configs.yaml")
seed = "NOT IMPLEMENTED"


class test_cnn:
    def __init__(
        self,
    ):
        self.name = args.model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = config[self.name]["training"]["lr"]
        self.test_loader = cnn_dataloaders.make_cnn_dataloader(
            data_args={
                "dataset_type": "test",
                "data_dir": args.data_dir,
                "metadata_dir": args.metadata_dir,
                "model_name": self.name,
            },
            dataset_class=DATASET_CLASSES[args.dataset],
            batch_size=config[self.name]["data"]["batch_size"],
        )

        num_classes = self.test_loader.dataset.get_num_classes()  # type: ignore as all the datasets have get_num_classes

        if self.name == "efficientnet":
            self.model = self._build_efficientnet(num_classes)

    def _build_efficientnet(self, num_classes):
        zero_shot = False
        if zero_shot:
            model = models.efficientnet_b0(weights="DEFAULT")
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)  # type: ignore as it is a sequential, able to be indexed
        else:
            model = models.efficientnet_b0()
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)  # type: ignore as it is a sequential, able to be indexed
            model.load_state_dict(
                torch.load(
                    os.path.join(args.weights_dir, f"{self.name}_fine_tuned.pt"),
                    map_location=self.device,
                )
            )

        model.eval()
        return model.to(self.device)

    def test(self):
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(
                self.test_loader, desc="Testing by classifying x number of images"
            ):
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(inputs)  # forward pass
                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f"{correct} / {total} correct\nAccuracy: {acc:.2f}%")

        os.makedirs("results/test_results.txt", exist_ok=True)
        with open("results/test_results.txt", "w") as out_file:
            out_file.write(
                f"Inference using seed: {seed} with\n{correct} / {total} correct\nAccuracy: {acc:.2f}%"
            )


def run_testing():
    new_train = test_cnn()
    new_train.test()


run_testing()
