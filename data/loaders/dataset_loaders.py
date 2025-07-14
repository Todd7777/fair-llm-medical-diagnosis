from torch.utils.data import DataLoader


class ChestXRayDataset:
    def __init__(
        self, data_dir, metadata_path, transform, split, demographic_key, label_col
    ):
        pass


class PathologyImageDataset:
    def __init__(
        self, data_dir, metadata_path, transform, split, demographic_key, label_col
    ):
        pass


class RetinalImageDataset:
    def __init__(
        self, data_dir, metadata_path, transform, split, demographic_key, label_col
    ):
        pass


def create_data_loaders(dataset, batch_size, num_workers=0, shuffle=False):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,  # Optional, can improve performance on GPU
    )
    return dataloader
