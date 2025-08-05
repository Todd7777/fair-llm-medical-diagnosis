from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

# Separated dataset wrappers for the distinct ordering of image and meta data
# transform is expected to be provided using dataset_maker. If None, no transform is applied.


# chexpert is already in a dataframe format
# data_dir in this context is the base directory of Chexpert, as Path contains the rest
class ChestXRayDataset(Dataset):
    def __init__(self, dataset_type, data_dir, metadata_dir, **kwargs):
        self.data_dir = data_dir

        if dataset_type == "train":
            self.metadata_file = "train.csv"
            self.split = None
        elif dataset_type == "eval":
            self.metadata_file = "valid_and_test.csv"
            self.split = "eval"
        elif dataset_type == "test":
            self.metadata_file = "valid_and_test.csv"
            self.split = "test"
        else:
            raise Exception('dataset types: "train", "eval", "test"')

        self.label_cols = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]

        metadata_path = os.path.join(metadata_dir, self.metadata_file)
        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
            print("csv exists")
        else:
            print("csv does not exist, creating")
            create_new_file(
                metadata_dir,
                self.metadata_file,
                self.label_cols + ["split"],
                ".csv",
                "_and_test.csv",
                {"eval": 0.5, "test": 0.5},
            )
            self.metadata = pd.read_csv(metadata_path)

        if self.split:
            self.metadata = self.metadata[
                self.metadata["split"] == self.split
            ].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_file_path = os.path.join(self.data_dir, row["Path"])
        image = Image.open(img_file_path).convert("RGB")
        label = row[self.label_cols].astype(float).values

        return {
            "image": image,
            "label": label,
        }

    def get_num_classes(self):
        return len(self.label_cols)


# data_dir in this context is the base directory of breakhis, as filename contains the rest
class PathologyImageDataset(Dataset):
    def __init__(self, dataset_type, data_dir, metadata_dir, **kwargs):
        self.data_dir = data_dir

        self.metadata_file = "Folds.csv"
        if dataset_type == "train":
            self.split = "train"
        elif dataset_type == "eval":
            self.split = "eval"
        elif dataset_type == "test":
            self.split = "test"
        else:
            raise Exception('dataset types: "train", "eval", "test"')

        metadata_path = os.path.join(metadata_dir, self.metadata_file)
        self.metadata = pd.read_csv(metadata_path)

        if self.split:
            self.metadata = self.metadata[
                self.metadata["grp"] == self.split
            ].reset_index(drop=True)

        self.labels = [
            "benign_adenosis",
            "malignant_adenosis",
            "benign_fibroadenoma",
            "malignant_fibroadenoma",
            "benign_phyllodes_tumor",
            "malignant_phyllodes_tumor",
            "benign_tubular_adenoma",
            "malignant_tubular_adenoma",
            "benign_ductal_carcinoma",
            "malignant_ductal_carcinoma",
            "benign_lobular_carcinoma",
            "malignant_lobular_carcinoma",
            "benign_mucinous_carcinoma",
            "malignant_mucinous_carcinoma",
            "benign_papillary_carcinoma",
            "malignant_papillary_carcinoma",
        ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_file_path = os.path.join(self.data_dir, row["Path"])
        image = Image.open(img_file_path).convert("RGB")

        path_list = row["filename"].split(os.sep)

        breast_idx = path_list.index("breast")
        sob_idx = path_list.index("SOB")
        benign_or_malignant = path_list[breast_idx + 1]
        class_name = f"{benign_or_malignant}_{path_list[sob_idx + 1]}"

        label = self.labels.index(class_name)
        return {
            "image": image,
            "label": label,
        }

    def get_num_classes(self):
        return len(self.labels)


# Subject to change based on how the retinal dataset's data is layed out
class RetinalImageDataset(Dataset):
    def __init__(self, dataset_type, data_dir, metadata_dir, **kwargs):
        super().__init__()
        self.data_dir = data_dir

        if dataset_type == "train":
            self.img_data_last_dir = "train"
            self.metadata_file = "train.csv"
        elif dataset_type == "eval":
            self.img_data_last_dir = "valid"
            self.metadata_file = "valid.csv"
        elif dataset_type == "test":
            self.img_data_last_dir = "test"
            self.metadata_file = "test.csv"
        else:
            raise Exception('dataset types: "train", "eval", "test"')

        metadata_path = os.path.join(metadata_dir, self.metadata_file)
        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
            print("csv exists")
        else:
            print("csv does not exist, creating")
            create_new_file(
                metadata_dir,
                self.metadata_file,
                ["Img_File_Name", "Label"],
                ".txt",
                ".csv",
            )
            self.metadata = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.metadata)

    # works when the keys are all at the top row, all the info following the same format in rows below; dataframe format
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_file_path = os.path.join(
            self.data_dir, self.img_data_last_dir, row["Img_File_Name"]
        )
        image = Image.open(img_file_path).convert("RGB")
        label = row["Label"]

        return {
            "image": image,
            "label": label,
        }

    def get_num_classes(self):
        return len(self.metadata["Label"].unique())


def create_new_file(
    metadata_dir, metadata_csv_file, names, file_type, replace_with, split_ratios=None
):
    txt_file = pd.read_csv(
        filepath_or_buffer=os.path.join(
            metadata_dir, metadata_csv_file.replace(replace_with, file_type)
        ),
        sep=" ",
        engine="python",
        header=None,
        names=names,
    )

    if split_ratios is None:
        pass
    else:
        # Ex: split_ratios = {'train': 0.8, 'eval': 0.1, 'test': 0.1}
        splits = list(split_ratios.keys())
        probs = list(split_ratios.values())

        np.random.seed(42)
        txt_file["split"] = np.random.choice(splits, size=len(txt_file), p=probs)

    txt_file.to_csv(os.path.join(metadata_dir, metadata_csv_file), index=False)


def create_data_loader(dataset, batch_size, num_workers, shuffle):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,  # Optional, can improve performance on GPU
    )
    return dataloader
