#!/usr/bin/env bash

# Fine tunes Efficientnet on , and tests
# From cnn directory, run ./experiments/train_and_test_efficinetnet_retina.sh

python train_cnn.py \
  --weights_dir path/store/weight/dir \
  --data_dir path/xxx/data/dir \
  --metadata_dir path/xxx/metadata/dir \
  --num_epochs 6 \
  --dataset retinal

python test_cnn.py \
  --weights_dir path/fine-tuned/weight/dir \
  --data_dir path/xxx/data/dir \
  --metadata_dir path/xxx/metadata/dir \
  --dataset chestxray retinal
