#

from tqdm import tqdm
import os
import yaml
import argparse

import torch
import torch.nn as nn
import torchvision.models as models
import cnn_dataset_maker
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import sys

sys.path.append("..")
import training_utils.early_stopping as early_stopping
from data.makedatasets.datasets import (
    RetinalImageDataset,
    ChestXRayDataset,
    PathologyImageDataset,
)

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
    parser.add_argument("--model_name", required=True, help="Model name n yaml config")
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

DATASET_CLASSES = {
    "retinal": RetinalImageDataset,
    "pathology": PathologyImageDataset,
    "chestxray": ChestXRayDataset,
}


# using adam as optimizing alg
# num workers should = num cpu threads(for data loading), currently at 4 workers, batches of 64
class TrainCnnDdp:
    def __init__(self, process_rank, world_size, use_cuda, args):
        self.args = args
        self.is_master = process_rank == 0
        self.use_cuda = use_cuda
        self.name = self.args.model_name
        self.device = torch.device(f"cuda:{process_rank}" if self.use_cuda else "cpu")

        if self.is_master:
            print("Using main device:", self.device)

        train_dataset = cnn_dataset_maker.make_cnn_dataset(
            data_args={
                "dataset_type": "train",
                "data_dir": self.args.data_dir,
                "metadata_dir": self.args.metadata_dir,
                "model_name": self.name,
            },
            dataset_class=DATASET_CLASSES[self.args.dataset],
        )
        eval_dataset = cnn_dataset_maker.make_cnn_dataset(
            data_args={
                "dataset_type": "eval",
                "data_dir": self.args.data_dir,
                "metadata_dir": self.args.metadata_dir,
                "model_name": self.name,
            },
            dataset_class=DATASET_CLASSES[self.args.dataset],
        )

        self.train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=process_rank, shuffle=True
        )
        self.eval_sampler = DistributedSampler(
            eval_dataset, num_replicas=world_size, rank=process_rank, shuffle=False
        )

        self.num_workers = args.num_workers

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config[self.name]["data"]["batch_size"],
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True if self.use_cuda else False,
        )
        self.eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=config[self.name]["data"]["batch_size"],
            sampler=self.eval_sampler,
            num_workers=self.num_workers,
            pin_memory=True if self.use_cuda else False,
        )

        self.num_epochs = config[self.name]["training"]["epochs"]
        self.lr = config[self.name]["training"]["lr"]
        self.criterion = nn.CrossEntropyLoss()  # If dataset is multiple diseases per image, use nn.BCEWithLogitsLoss instead of nn.CrossEntropyLoss
        self.weight_decay = (
            config[self.name]["training"]["weight_decay"] if not None else 0
        )
        self.warmup_steps = (
            config[self.name]["training"]["warmup_steps"] if not None else 0
        )
        self.optimizer = None
        self.warmup_scheduler = None
        self.cosine_scheduler = None

        os.makedirs(self.args.weights_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.args.weights_dir, f"{self.name}_{self.args.dataset}_fine_tuned_best.pt"
        )
        num_classes = self.train_loader.dataset.get_num_classes()  # type: ignore as all the datasets have get_num_classes
        if self.name == "efficientnet_v2":
            self.model = self._build_efficientnet_v2(num_classes)
            self.early_stopping = early_stopping.EarlyStopping(
                patience=4, path=checkpoint_path, is_master=self.is_master
            )
        elif self.name == "densenet":
            self.model = self._build_densenet(num_classes)
            self.early_stopping = early_stopping.EarlyStopping(
                patience=6, path=checkpoint_path, is_master=self.is_master
            )
        elif self.name == "convnext":
            self.model = self._build_convnext(num_classes)
            self.early_stopping = early_stopping.EarlyStopping(
                patience=6, path=checkpoint_path, is_master=self.is_master
            )
        else:
            raise Exception("wrong model name")

        if self.use_cuda:
            self.model = DDP(self.model, device_ids=[process_rank])
        else:
            self.model = DDP(self.model)  # no device_ids for CPU

        if "warmup_steps" in config[self.name]["training"]:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,  # type: ignore as will always be instantiated
                start_factor=0.1,
                total_iters=self.warmup_steps,
            )
        if "cosine_annealing" in config[self.name]["training"]:
            self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,  # type: ignore
                T_max=config[self.name]["training"]["cosine_annealing"]["T_max"],
                eta_min=config[self.name]["training"]["cosine_annealing"]["eta_min"],
            )

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

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return model.to(self.device)

    def save_model(self, process_rank):
        os.makedirs(self.args.weights_dir, exist_ok=True)
        path = os.path.join(
            self.args.weights_dir,
            f"{self.name}_{self.args.dataset}_fine_tuned_last_epoch.pt",
        )
        if process_rank == 0:
            model_to_save = (
                self.model.module
                if hasattr(self.model, "module")
                else self.model  # gpu has attr module
            )
            torch.save(model_to_save.state_dict(), path)  # type: ignore
            print(f"Model saved to {path}")

    def train(self):
        model_to_pass = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        os.makedirs("results", exist_ok=True)
        out_file = open(
            os.path.join(
                "results", f"{self.name}_{args.dataset}_train_results.txt", "w"
            )
        )
        out_file.write(f"Training using seed: {seed}\n")

        num_epochs = config[self.name]["training"]["epochs"]

        self.model.train()

        for epoch in range(num_epochs):
            self.train_sampler.set_epoch(epoch)
            self.eval_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            correct = 0
            total = 0

            if self.warmup_scheduler is not None and self.warmup_steps > epoch:
                self.warmup_scheduler.step()  # type: ignore
            elif self.warmup_scheduler is not None:
                self.cosine_scheduler.step()  # type: ignore

            current_lr = self.optimizer.param_groups[0]["lr"]  # type: ignore
            print(f"Learning rate at epoch {epoch + 1}: {current_lr:.6f}")

            tqdm_iterator = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                disable=not self.is_master,
            )

            for batch in tqdm_iterator:
                inputs = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                outputs = self.model(inputs)  # forward pass
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()  # type: ignore
                loss.backward()
                self.optimizer.step()  # type: ignore

                batch_size = labels.size(0)
                epoch_loss += (
                    loss.item() * batch_size
                )  # sum loss weighted by batch size
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += batch_size

            # Convert to tensors for distributed reduction
            loss_tensor = torch.tensor(epoch_loss, device=self.device)
            correct_tensor = torch.tensor(correct, device=self.device)
            total_tensor = torch.tensor(total, device=self.device)

            # Sum across all GPUs/processes
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

            avg_loss = loss_tensor.item() / total_tensor.item()
            accuracy = 100 * correct_tensor.item() / total_tensor.item()

            if self.is_master:
                print(
                    f"Epoch {epoch + 1}: Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%"
                )
                out_file.write(
                    f"Epoch {epoch + 1}: Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%\n"
                )

            val_loss = self.validate(
                out_file
            )  # should also modify validate similarly for aggregation

            self.early_stopping(val_loss, model_to_pass)
            stop_flag = torch.tensor(0, device=self.device)
            if self.early_stopping.early_stop:
                stop_flag += 1

            dist.all_reduce(stop_flag, op=dist.ReduceOp.SUM)

            if stop_flag.item() > 0:
                print(f"Process rank {dist.get_rank()} stopping early.")
                break

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

        out_file.close()
        if self.early_stopping.early_stop is False:
            self.early_stopping.save_checkpoint(
                self.early_stopping.val_loss_best, model_to_pass, final_save=True
            )

    def validate(self, out_file):
        self.model.eval()
        correct = torch.tensor(0, device=self.device)
        total = torch.tensor(0, device=self.device)
        valid_loss_sum = torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            for batch in self.eval_loader:
                inputs = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                batch_size = labels.size(0)
                valid_loss_sum += loss.item() * batch_size
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum()
                total += batch_size

        dist.all_reduce(valid_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

        acc = 100 * correct.item() / total.item()
        avg_loss = valid_loss_sum.item() / total.item()

        if self.is_master:
            print(f"Validation Accuracy: {acc:.2f}%")
            out_file.write(f"Validation Accuracy: {acc:.2f}%")
            out_file.write(f"Validation Loss per batch: {avg_loss}\n\n")
        self.model.train()
        return avg_loss


def main_worker(rank, world_size, args):
    set_seed(seed + rank)

    use_cuda = torch.cuda.is_available() and args.total_gpus is not None
    if use_cuda:
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl" if use_cuda else "gloo",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    trainer = TrainCnnDdp(rank, world_size, use_cuda, args)
    trainer.train()
    trainer.save_model(rank)

    dist.destroy_process_group()


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

    world_size = len(available_gpus)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # one process per GPU
    mp.spawn(  # type: ignore
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
