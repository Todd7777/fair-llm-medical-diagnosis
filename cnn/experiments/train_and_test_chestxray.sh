#!/usr/bin/env bash

# From cnn directory, run ./experiments/train_and_test_chestxray.sh

printf "Below is the usage and description for each parameter of train_cnn:\n"
python train_cnn.py --help
printf "\nBelow is the usage and description for each parameter of test_cnn:\n"
python test_cnn.py --help

echo "Enter weights directory:"
read -r weights_dir

echo "Enter data directory:"
read -r data_dir
if [ ! -d "$data_dir" ]; then
  echo "Error: Data directory '$data_dir' does not exist."
  exit 1
fi

echo "Enter metadata directory:"
read -r metadata_dir
if [ ! -d "$metadata_dir" ]; then
  echo "Error: Metadata directory '$metadata_dir' does not exist."
  exit 1
fi


echo "Enter CNN model name:"
read -r model_name

# Define valid model names
valid_models="efficientnet_v2 densenet convnext"

# Check if input is in valid_models list
if ! [[ " $valid_models " =~ " $model_name " ]]; then
  echo "Error: model name does not exist. Names: efficientnet_v2, densenet, convnext"
  exit 1
fi


printf "\nRunning training...\n"
python3 train_cnn.py \
  --weights_dir "$weights_dir" \
  --data_dir "$data_dir" \
  --metadata_dir "$metadata_dir" \
  --model_name "$model_name" \
  --dataset "chestxray"

printf "\nRunning testing...\n"
python3 test_cnn.py \
  --weights_dir "$weights_dir" \
  --data_dir "$data_dir" \
  --metadata_dir "$metadata_dir" \
  --model_name "$model_name" \
  --dataset "chestxray"
