from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset

# Separated dataset wrappers for the distinct ordering of image and meta data


# chexpert is already in a dataframe format
class ChestXRayDataset(Dataset):
    def __init__(
        self,
        data_dir,
        metadata_path,
        transform=None,
        split=None,
        demographic_key="Sex",
        label_cols=None,
    ):
        self.metadata = pd.read_csv(metadata_path)
        if split:
            self.metadata = self.metadata[self.metadata["split"] == split]
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.demographic_key = demographic_key
        self.label_cols = label_cols or [
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
            "No Finding",
        ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.data_dir, row["Path"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = row[self.label_cols].astype(float).values
        demographic = row.get(self.demographic_key, "unknown")

        return {
            "image": image,
            "label": label,
            "demographic": demographic,
            "path": row["Path"],
        }

    def get_num_classes(self):
        pass


class PathologyImageDataset(Dataset):
    def __init__(self, data_dir, metadata_dir, transform, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.metadata_dir = metadata_dir
        self.transform = transform

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def get_num_classes(self):
        pass


# Subject to change based on how the retinal dataset's data is layed out
class RetinalImageDataset(Dataset):
    def __init__(self, dataset_type, data_dir, metadata_dir, transform, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
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

        if os.path.exists(os.path.join(metadata_dir, self.metadata_file)):
            self.metadata = pd.read_csv(os.path.join(metadata_dir, self.metadata_file))
            print("csv exists")
        else:
            print("csv does not exist, creating")
            convert_to_csv(metadata_dir, self.metadata_file, ".txt")

    def __len__(self):
        return len(self.metadata)

    # works when the keys are all at the top row, all the info following the same format in rows below; dataframe format
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_file_path = os.path.join(
            self.data_dir, self.img_data_last_dir, row["Img_File_Name"]
        )
        image = Image.open(img_file_path)
        if self.transform:
            image = self.transform(image)  # PIL image
        label = row["Label"]

        return {
            "image": image,
            "label": label,
        }

    def get_num_classes(self):
        return len(self.metadata["Label"].unique())


def convert_to_csv(metadata_dir, metadata_csv_file, file_type):
    txt_file = pd.read_csv(
        filepath_or_buffer=os.path.join(
            metadata_dir, metadata_csv_file.replace(".csv", file_type)
        ),
        sep=" ",
        engine="python",
        header=None,
        names=["Img_File_Name", "Label"],
    )
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
