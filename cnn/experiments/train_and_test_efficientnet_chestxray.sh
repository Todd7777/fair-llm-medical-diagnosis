#!/usr/bin/env bash

# Fine tunes Efficientnet on chexpert, and tests
# From cnn directory, run ./experiments/train_and_test_efficinetnet_chestxray.sh

python train_cnn.py \
  --weights_dir path/store/weight/dir \
  --data_dir path/chexpert/data/dir \
  --metadata_dir path/chexpert/metadata/dir \
  --num_epochs 6 \
  --dataset chestxray

python test_cnn.py \
  --weights_dir path/fine-tuned/weight/dir \
  --data_dir path/chexpert/data/dir \
  --metadata_dir path/chexpert/metadata/dir \
  --dataset chestxray
