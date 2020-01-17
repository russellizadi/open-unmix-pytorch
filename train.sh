#!/bin/bash

# choose the cuda id
export CUDA_VISIBLE_DEVICES=0
echo CUDA ID: $CUDA_VISIBLE_DEVICES

# set args
dataset="musdb"

name_dataset="musdb18_debug"

name_model="OpenUnmixHidden"
#name_model="OpenUnmixMixup"

tag_model="umh_epoch10"

epochs="10"
targets="vocals"
#targets="vocals drums bass other"

# train
root="/home/russell/russellizadi/datasets/$dataset/$name_dataset"
output="models/$tag_model"
is_wav="--is-wav"

for target in $targets; do
	python train.py --dataset $dataset --root=$root --name-model $name_model --epochs $epochs --output=$output --target=$target --is-wav
done
