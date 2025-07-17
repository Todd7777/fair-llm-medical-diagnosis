from torchvision import transforms

import sys

sys.path.append("..")
import data.makedatasets.datasets as datasets


# can split up into 2 dirs for train and val
def make_cnn_dataloaders(data_dir, metadata_path, batch_size):
    transform = transforms.Compose(  # img preprocess pipeline
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(),   # maybe
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # common values for NNs trained on imagenet
        ]
    )

    # Any PyTorch Dataset class that implements the __getitem__ and __len__ methods can be passed into a DataLoader
    # Maybe i should use ImageFolder if it works

    train_dataset = datasets.RetinalImageDataset(
        data_dir,
        metadata_path,
        transform,
        split=None,
    )
    eval_dataset = None

    train_loader = datasets.create_data_loader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    eval_loader = datasets.create_data_loader(
        dataset=eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return train_loader, eval_loader
