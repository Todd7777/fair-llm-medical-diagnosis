from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset

# If we establish exactly the format of the data, and they are the same across diff. datasets, they can be the same subclass


class ChestXRayDataset(Dataset):
    def __init__(
        self, data_dir, metadata_path, transform, split, demographic_key, label_col
    ):
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.transform = transform
        self.split = split
        self.demographic_key = demographic_key
        self.label_col = label_col

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class PathologyImageDataset(Dataset):
    def __init__(
        self, data_dir, metadata_path, transform, split, demographic_key, label_col
    ):
        super().__init__()
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.transform = transform
        self.split = split
        self.demographic_key = demographic_key
        self.label_col = label_col

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class RetinalImageDataset(Dataset):
    def __init__(
        self, data_dir, metadata_path, transform, split, demographic_key, label_col
    ):
        super().__init__()
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_path)
        self.transform = transform
        self.split = split
        self.demographic_key = demographic_key
        self.label_col = label_col

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.data_dir, row["path"])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)  # PIL image
        label = row[self.label_col]
        demographic = row[self.demographic_key]  # This should mean column

        return {"image": image, "label": label, "demographic": demographic}


def create_data_loaders(dataset, batch_size, num_workers=0, shuffle=False):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,  # Optional, can improve performance on GPU
    )
    return dataloader
