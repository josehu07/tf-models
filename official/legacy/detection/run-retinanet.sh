#!/usr/bin/bash

MODE=$1

python3 main.py \
  --model_dir=./retinanet_model \
  --strategy_type=one_device \
  --num_gpus=0 \
  --mode=$MODE \
  --params_override="eval:
 eval_file_pattern: retinanet_data/val.tfrecord
 batch_size: 2
 val_json_file: retinanet_data/annotations/instances_val2017.json
 eval_samples: 2
train:
 total_steps: 0
 batch_size: 8
 train_file_pattern: retinanet_data/train.tfrecord
use_tpu: False
"

