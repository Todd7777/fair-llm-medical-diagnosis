from torchvision import transforms
import yaml
import sys
from pathlib import Path

# can split up into 2 dirs for train and val
def make_cnn_dataset(data_args, dataset_class):
    # Project structure to add cnn configs
    full_path = Path(sys.argv[0]).resolve()
    two_levels_up = full_path.parents[1]
    config_path = two_levels_up.joinpath("cnn").joinpath("cnn_configs.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_args = dict(data_args)  # doesn't edit orig
    
    dataset_type = data_args["dataset_type"]
    dataset_name = data_args["dataset_name"]
    model_name = data_args["model_name"]
    
    transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]

    if dataset_type == "train" and config[model_name][dataset_name]["training"]["geo_transformation"] == "rand_horiz_flip":
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # common values for weights trained on imagenet: efficientnetv2, convnext, and densenet
    ])

    transform = transforms.Compose(transform_list)



    data_args["transform"] = transform

    dataset = dataset_class(**data_args)

    return dataset


def make_vlm_dataset(data_args, dataset_class):
    transform = None

    data_args = dict(data_args)  # doesn't edit orig

    data_args["transform"] = transform

    dataset = dataset_class(**data_args)

    pass
    # return dataset
