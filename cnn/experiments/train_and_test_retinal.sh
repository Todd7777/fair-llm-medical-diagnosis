#!/usr/bin/env bash

# TODO: Add selection of CNN model with parseargs and implment it here
# currently only uses efficientnet

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

printf "\nRunning training...\n"
python train_cnn.py \
  --weights_dir "$weights_dir" \
  --data_dir "$data_dir" \
  --metadata_dir "$metadata_dir" \
  --model_name "efficientnet" \
  --num_epochs 6 \
  --dataset "retinala"

printf "\nRunning testing...\n"
python test_cnn.py \
  --weights_dir "$weights_dir" \
  --data_dir "$data_dir" \
  --metadata_dir "$metadata_dir" \
  --model_name "efficientnet" \
  --dataset "retinal"
