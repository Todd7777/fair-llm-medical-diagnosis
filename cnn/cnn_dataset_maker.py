from torchvision import transforms


# can split up into 2 dirs for train and val
def make_cnn_dataset(data_args, dataset_class):
    transform = None
    if data_args["model_name"] == "efficientnet_v2":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),  # maybe
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # common values for NNs trained on imagenet
            ]
        )
    elif data_args["model_name"] == "densenet":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),  # maybe
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # common values for NNs trained on imagenet
            ]
        )

    elif data_args["model_name"] == "convnext":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),  # maybe
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # common values for NNs trained on imagenet
            ]
        )

    data_args = dict(data_args)  # doesn't edit orig

    data_args["transform"] = transform

    dataset = dataset_class(**data_args)

    return dataset

    """
    return 
    datasets.create_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=(data_args["dataset_type"] == "train"),
        num_workers=4,
    )"""
