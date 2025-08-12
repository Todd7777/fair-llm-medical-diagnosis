# using fine tuned model weights

import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml
from tqdm import tqdm
import argparse
import os
import sys

sys.path.append("..")
from data.makedatasets.datasets import (
    RetinalImageDataset,
    ChestXRayDataset,
    PathologyImageDataset,
)
from data.makedatasets import dataset_maker

sys.path.remove("..")


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


DATASET_CLASSES = {
    "retinal": RetinalImageDataset,
    "pathology": PathologyImageDataset,
    "chestxray": ChestXRayDataset,
}


args = parse_args()

if args.gpu is not None:
    torch.cuda.set_device(int(args.gpu))
elif torch.cuda.is_available():
    torch.cuda.set_device(0)

config = load_config("cnn_configs.yaml")


class TestCnn:
    def __init__(
        self,
    ):
        self.name = args.model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.dataset_name = args.dataset
        self.lr = config[self.name][self.dataset_name]["training"]["lr"]
        self.num_workers = args.num_workers

        test_dataset = dataset_maker.make_cnn_dataset(
            data_args={
                "dataset_type": "test",
                "data_dir": args.data_dir,
                "metadata_dir": args.metadata_dir,
                "model_name": self.name,
            },
            dataset_class=DATASET_CLASSES[self.dataset_name],
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config[self.name][self.dataset_name]["data"]["batch_size"],
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

        num_classes = self.test_loader.dataset.get_num_classes()  # type: ignore as all the datasets have get_num_classes

        if self.name == "efficientnet_v2":
            self.model = self._build_efficientnet_v2(num_classes)
        elif self.name == "densenet":
            self.model = self._build_densenet(num_classes)
        elif self.name == "convnext":
            self.model = self._build_convnext(num_classes)

    def _build_efficientnet_v2(self, num_classes):
        zero_shot = False
        if zero_shot:
            model = models.efficientnet_v2_m(weights="DEFAULT")
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)  # type: ignore as it is a sequential, able to be indexed
        else:
            model = models.efficientnet_v2_m()
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)  # type: ignore as it is a sequential, able to be indexed
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        args.weights_dir,
                        f"{self.name}_{args.dataset}_fine_tuned_best.pt",
                    ),
                    map_location=self.device,
                )
            )

        model.eval()
        return model.to(self.device)

    def _build_densenet(self, num_classes):
        zero_shot = False
        if zero_shot:
            model = models.densenet121(weights="DEFAULT")
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)  # type: ignore as it is a sequential, able to be indexed
        else:
            model = models.densenet121()
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)  # type: ignore as it is a sequential, able to be indexed
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        args.weights_dir,
                        f"{self.name}_{args.dataset}_fine_tuned_best.pt",
                    ),
                    map_location=self.device,
                )
            )

        model.eval()
        return model.to(self.device)

    def _build_convnext(self, num_classes):
        zero_shot = False
        if zero_shot:
            model = models.convnext_tiny(weights="DEFAULT")
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, num_classes)  # type: ignore as it is a sequential, able to be indexed
        else:
            model = models.convnext_tiny()
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, num_classes)  # type: ignore as it is a sequential, able to be indexed
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        args.weights_dir,
                        f"{self.name}_{args.dataset}_fine_tuned_best.pt",
                    ),
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
                self.test_loader,
                desc=f"Testing by classifying {len(self.test_loader.dataset)} number of images",  # type: ignore
            ):
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(inputs)  # forward pass
                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print("\nCuda memory allocated (GB):", torch.cuda.memory_allocated() / 1024**3)
        print(
            "Cuda max memory reserved (GB):",
            torch.cuda.max_memory_reserved() / 1024**3,
            "\n",
        )
        acc = 100 * correct / total
        print(f"{correct} / {total} correct\nAccuracy: {acc:.2f}%")

        os.makedirs("results", exist_ok=True)
        with open(
            os.path.join("results", f"{self.name}_{args.dataset}_train_results.txt"),
            "w",
        ) as out_file:
            out_file.write(
                f"Inference with\n{correct} / {total} correct\nAccuracy: {acc:.2f}%"
            )


def main():
    try:
        tester = TestCnn()
        tester.test()
    except KeyboardInterrupt:
        print("Testing interrupted.")
    finally:
        torch.cuda.empty_cache()
        print("Cleanup done, exiting.")


if __name__ == "__main__":
    main()
