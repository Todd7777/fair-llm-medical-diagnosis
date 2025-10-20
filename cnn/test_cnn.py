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

import torch.nn.functional as F
from sklearn.metrics import roc_curve
from torchmetrics.classification import (
    Accuracy, MulticlassF1Score, BinaryF1Score, MultilabelF1Score,
    MulticlassAUROC, BinaryAUROC, MultilabelAUROC,
    MulticlassAveragePrecision, BinaryAveragePrecision, MultilabelAveragePrecision,
    CalibrationError
)

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
                "model_name": self.name,
                "dataset_name": self.dataset_name,
                "data_dir": args.data_dir,
                "metadata_dir": args.metadata_dir,
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

        self.classification_type = test_dataset.get_classification_type()

        self.num_classes = self.test_loader.dataset.get_num_classes()  # type: ignore as all the datasets have get_num_classes

        if self.name == "efficientnet_v2":
            self.model = self._build_efficientnet_v2()
        elif self.name == "densenet":
            self.model = self._build_densenet()
        elif self.name == "convnext":
            self.model = self._build_convnext()

    def _build_efficientnet_v2(self):
        zero_shot = False
        if zero_shot:
            model = models.efficientnet_v2_m(weights="DEFAULT")
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.num_classes)  # type: ignore as it is a sequential, able to be indexed
        else:
            model = models.efficientnet_v2_m()
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.num_classes)  # type: ignore as it is a sequential, able to be indexed
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        args.weights_dir,
                        f"{self.name}_{self.dataset_name}_fine_tuned_best.pt",
                    ),
                    map_location=self.device,
                )
            )

        model.eval()
        return model.to(self.device)

    def _build_densenet(self):
        zero_shot = False
        if zero_shot:
            model = models.densenet121(weights="DEFAULT")
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)  # type: ignore as it is a sequential, able to be indexed
        else:
            model = models.densenet121()
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)  # type: ignore as it is a sequential, able to be indexed
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        args.weights_dir,
                        f"{self.name}_{self.dataset_name}_fine_tuned_best.pt",
                    ),
                    map_location=self.device,
                )
            )

        model.eval()
        return model.to(self.device)

    def _build_convnext(self):
        zero_shot = False
        if zero_shot:
            model = models.convnext_tiny(weights="DEFAULT")
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, self.num_classes)  # type: ignore as it is a sequential, able to be indexed
        else:
            model = models.convnext_tiny()
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, self.num_classes)  # type: ignore as it is a sequential, able to be indexed
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        args.weights_dir,
                        f"{self.name}_{self.dataset_name}_fine_tuned_best.pt",
                    ),
                    map_location=self.device,
                )
            )

        model.eval()
        return model.to(self.device)

    def test(self):
        all_probs = []
        all_preds = []
        all_labels = []
        all_outputs = []

        if self.classification_type == "multi_class":
            accuracy_metric = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
            f1_metric = MulticlassF1Score(num_classes=self.num_classes, average='macro').to(self.device)
            auc_metric = MulticlassAUROC(num_classes=self.num_classes, average='macro').to(self.device)
            auprc_metric = MulticlassAveragePrecision(num_classes=self.num_classes, average="macro").to(self.device)
            ece_metric = CalibrationError(task="multiclass", num_classes=self.num_classes, n_bins=15).to(self.device)

        elif self.classification_type == "binary":
            accuracy_metric = Accuracy(task="binary").to(self.device)
            f1_metric = BinaryF1Score().to(self.device)
            auc_metric = BinaryAUROC().to(self.device)
            auprc_metric = BinaryAveragePrecision().to(self.device)
            ece_metric = CalibrationError(task="binary", n_bins=15).to(self.device)

        elif self.classification_type == "multi_label":
            accuracy_metric = Accuracy(task="multilabel", num_labels=self.num_classes).to(self.device)
            f1_metric = MultilabelF1Score(num_labels=self.num_classes, average='macro').to(self.device)
            auc_metric = MultilabelAUROC(num_labels=self.num_classes, average='macro').to(self.device)
            auprc_metric = MultilabelAveragePrecision(num_labels=self.num_classes, average="macro").to(self.device)
            ece_metric = None
        else:
            raise Exception("Not a valid classification type")

        with torch.no_grad():
            for batch in tqdm(
                self.test_loader,
                desc=f"Testing by classifying {len(self.test_loader.dataset)} number of images",  # type: ignore
            ):
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(inputs)  # forward pass
 
                if self.classification_type == "binary":
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).long()
                elif self.classification_type == "multi_class":
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                elif self.classification_type == "multi_label":
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                else:
                    raise Exception("Not a type of classification")
       
                preds = preds.to(self.device)
                labels = labels.to(self.device)

                all_probs.append(probs)
                all_preds.append(preds)
                all_labels.append(labels)
                all_outputs.append(outputs)
                

        print("\nCuda memory allocated (GB):", torch.cuda.memory_allocated() / 1024**3)
        print(
            "Cuda max memory reserved (GB):",
            torch.cuda.max_memory_reserved() / 1024**3,
            "\n",
        )
        
        all_probs = torch.cat(all_probs)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_outputs = torch.cat(all_outputs)

        if self.classification_type == "binary" or self.classification_type == "multi_class":
            unique, counts = torch.unique(all_labels, return_counts=True)
            print("Class distribution in test set:", dict(zip(unique.tolist(), counts.cpu().tolist())))
        elif self.classification_type == "multi_label":
            class_counts = torch.sum(all_labels, dim=0).to(torch.int)
            print("Class distribution in test set:", class_counts.cpu().tolist())
        
        all_labels = all_labels.to(torch.long)
        accuracy_metric.update(all_preds, all_labels)
        auprc_metric.update(all_probs, all_labels)
        auc_metric.update(all_probs, all_labels)
        f1_metric.update(all_preds, all_labels)

        acc = accuracy_metric.compute() * 100
        accuracy_metric.reset()
        print(f"Accuracy: {acc:.2f}%")
        macro_auc = auc_metric.compute()
        auc_metric.reset()
        print(f"Macro-AUC: {macro_auc:.4f}")
        macro_auprc = auprc_metric.compute()
        auprc_metric.reset()
        print(f"Macro-AUPRC: {macro_auprc:.4f}")
        f1_score = f1_metric.compute()
        f1_metric.reset()
        print(f"Macro F1 Score: {f1_score:.4f}")

        if self.classification_type == "multi_class":
            nll = F.cross_entropy(all_outputs, all_labels, reduction="mean").item()
        elif self.classification_type == "binary" or self.classification_type == "multi_label":
            nll = F.binary_cross_entropy_with_logits(all_outputs, all_labels.float(), reduction="mean").item()
        else:
            raise Exception("Not a valid classification type")
        print(f"Negative Log-Likelihood: {nll:.4f}")

        if self.classification_type == "binary":
            all_probs_np = all_probs.numpy().flatten()
            all_labels_np = all_labels.numpy().flatten()

            fpr, tpr, thresholds = roc_curve(all_labels_np, all_probs_np)
            specificity = 1 - fpr
            idx = (np.abs(specificity - 0.90)).argmin()
            sensitivity_at_90_specificity = tpr[idx]
            print(f"Sensitivity at 90% specificity: {sensitivity_at_90_specificity:.4f}")

            ece_metric.update(all_probs, all_labels)
            ece = ece_metric.compute()
            ece_metric.reset()
            print(f"ECE: {ece:4f}")
        elif self.classification_type == "multi_class":
            sensitivity_at_90_specificity = "Only available for binary classification" 
            print(f"Sensitivity at 90% specificity: {sensitivity_at_90_specificity}")

            ece_metric.update(all_probs, all_labels)
            ece = ece_metric.compute()
            ece_metric.reset()
            print(f"ECE: {ece:4f}")
        elif self.classification_type == "multi_label":
            sensitivity_at_90_specificity = "Only available for binary classification" 
            print(f"Sensitivity at 90% specificity: {sensitivity_at_90_specificity}")
        
            ece = "Not available for multi label classification"
            print(f"ECE: {ece}")

        if self.classification_type == "multi_class":
            labels_one_hot = torch.nn.functional.one_hot(all_labels.long(), num_classes=all_probs.shape[1]).float()
            brier_score = torch.mean((all_probs - labels_one_hot) ** 2).item()
        else:
            brier_score = torch.mean((all_probs - all_labels.float()) ** 2).item()
        print(f"Brier Score: {brier_score:.4f}")

        os.makedirs("results", exist_ok=True)
        with open(
            os.path.join(
                "results", f"{self.name}_{self.dataset_name}_train_results.txt"
            ),
            "w",
        ) as out_file:
            out_file.write(f"Accuracy: {acc:.2f}%\n") 
            out_file.write(f"Macro-AUC: {macro_auc:.4f}\n")
            out_file.write(f"Macro-AUPRC: {macro_auprc:.4f}\n")
            out_file.write(f"Sensitivity at 90% specificity: {sensitivity_at_90_specificity}\n")
            out_file.write(f"Brier Score: {brier_score:.4f}\n")
            out_file.write(f"ECE: {ece}\n")
            out_file.write(f"Negative Log-Likelihood: {nll:.4f}\n")
            out_file.write(f"Macro F1 Score: {f1_score:.4f}\n")

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
