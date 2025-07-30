# using fine tuned model weights

import torch
import torchvision.models as models
import torch.nn as nn
import cnn_dataset_maker
import yaml
from tqdm import tqdm
import argparse
import os
from data.makedatasets.datasets import (
    RetinalImageDataset,
    ChestXRayDataset,
    PathologyImageDataset,
)

import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


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
        "--zero_shot", required=False, default="False", help='"True, False"'
    )
    return parser.parse_args()


args = parse_args()
if (
    args.zero_shot == "False" or args.zero_shot is not None
) and args.weights_dir is None:
    raise Exception("Must input args for either --zero_shot or --weights_dir")

DATASET_CLASSES = {
    "retinal": RetinalImageDataset,
    "pathology": PathologyImageDataset,
    "chestxray": ChestXRayDataset,
}


config = load_config("cnn_configs.yaml")
seed = "NOT IMPLEMENTED"


class TestCnn:
    def __init__(self, process_rank, world_size, use_cuda):
        self.use_cuda = use_cuda
        self.name = args.model_name
        self.device = torch.device(f"cuda:{process_rank}" if self.use_cuda else "cpu")
        print("Using device:", self.device)
        self.lr = config[self.name]["training"]["lr"]
        test_dataset = cnn_dataset_maker.make_cnn_dataset(
            data_args={
                "dataset_type": "test",
                "data_dir": args.data_dir,
                "metadata_dir": args.metadata_dir,
                "model_name": self.name,
            },
            dataset_class=DATASET_CLASSES[args.dataset],
        )

        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=process_rank, shuffle=False
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config[self.name]["data"]["batch_size"],
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True if self.use_cuda else False,
        )

        num_classes = self.test_loader.dataset.get_num_classes()  # type: ignore as all the datasets have get_num_classes

        if self.name == "efficientnet_v2":
            self.model = self._build_efficientnet_v2(num_classes)
        elif self.name == "densenet":
            self.model = self._build_densenet(num_classes)
        elif self.name == "convnext":
            self.model = self._build_convnext(num_classes)
        else:
            raise Exception("wrong model name")

        if self.use_cuda:
            self.model = DDP(self.model, device_ids=[process_rank])
        else:
            self.model = DDP(self.model)  # no device_ids for CPU

        if args.zero_shot == "True":
            self.zero_shot = True
        elif args.zero_shot == "False":
            self.zero_shot = False
        else:
            raise Exception('Argument for --zero_shot must be either "True" or "False"')

    def _build_efficientnet_v2(self, num_classes):
        if self.zero_shot:
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
                        args.weights_dir, f"{self.name}_{args.dataset}_fine_tuned.pt"
                    ),
                    map_location=self.device,
                )
            )

        model.eval()
        return model.to(self.device)

    def _build_densenet(self, num_classes):
        if self.zero_shot:
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
                        args.weights_dir, f"{self.name}_{args.dataset}_fine_tuned.pt"
                    ),
                    map_location=self.device,
                )
            )

        model.eval()
        return model.to(self.device)

    def _build_convnext(self, num_classes):
        if self.zero_shot:
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
                        args.weights_dir, f"{self.name}_{args.dataset}_fine_tuned.pt"
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
                self.test_loader, desc="Testing by classifying x number of images"
            ):
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(inputs)  # forward pass
                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        if self.use_cuda:
            print(
                "\nCuda memory allocated (GB):",
                torch.cuda.memory_allocated() / 1024**3,
            )
            print(
                "Cuda max memory reserved (GB):",
                torch.cuda.max_memory_reserved() / 1024**3,
                "\n",
            )

        acc = 100 * correct / total
        print(f"{correct} / {total} correct\nAccuracy: {acc:.2f}%")

        os.makedirs("results", exist_ok=True)
        with open("results/test_results.txt", "w") as out_file:
            out_file.write(
                f"Inference using seed: {seed} with\n{correct} / {total} correct\nAccuracy: {acc:.2f}%"
            )


def ddp_setup(process_rank, world_size, use_cuda):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    backend = "nccl" if use_cuda else "gloo"
    if use_cuda:
        torch.cuda.set_device(process_rank)
    init_process_group(backend=backend, rank=process_rank, world_size=world_size)


def main(process_rank, world_size, use_cuda):
    ddp_setup(process_rank, world_size, use_cuda)
    tester = TestCnn(process_rank, world_size, use_cuda)
    tester.test()
    destroy_process_group()


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_cuda else 1
    mp.spawn(main, args=(world_size, use_cuda), nprocs=world_size)  # type: ignore
