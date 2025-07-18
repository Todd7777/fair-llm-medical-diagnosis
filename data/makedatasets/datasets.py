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
    def __init__(self, data_dir, metadata_path, transform):
        super().__init__()
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.transform = transform

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def get_num_classes(self):
        pass


# Subject to change based on how the retinal dataset's data is layed out
class RetinalImageDataset(Dataset):
    def __init__(self, dataset_type, data_dir, metadata_path, transform):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        if dataset_type == "train":
            self.img_data_last_dir = "whatevertrainlastdir"
            self.metadata_file = "whatevertrainmetadatafile"
        elif dataset_type == "eval":
            self.img_data_last_dir = "whateverevallastdir"
            self.metadata_file = "whateverevalmetadatafile"
        elif dataset_type == "test":
            self.img_data_last_dir = "whatevertestlastdir"
            self.metadata_file = "whatevertestmetadatafile"
        else:
            raise Exception('dataset types: "train", "eval", "test"')

        self.metadata = pd.read_csv(os.path.join(metadata_path, self.metadata_file))

    def __len__(self):
        return len(self.metadata)

    # works when the keys are all at the top row, all the info following the same format in rows below; dataframe format
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.data_dir, self.img_data_last_dir, row["Path"])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)  # PIL image
        label = row["label"]
        demographic = row["demographic"]

        return {
            "image": image,
            "label": label,
            # "demographic": demographic, # won't be used
        }  # def something diff

    def get_num_classes(self):
        return len(self.metadata["label"].unique())


def create_data_loader(dataset, batch_size, num_workers, shuffle):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,  # Optional, can improve performance on GPU
    )
    return dataloader
