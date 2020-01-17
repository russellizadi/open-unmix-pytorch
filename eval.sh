#!/bin/bash

# choose the cuda id
export CUDA_VISIBLE_DEVICES=0
echo CUDA ID: $CUDA_VISIBLE_DEVICES

# set args
path_datasets="/home/russell/russellizadi/datasets"
dataset="musdb"
name_dataset="musdb18_debug"
epochs="1"
targets="vocals"
#targets="vocals drums bass other"
samples_per_track="64"
is_wav="--is-wav"

# don't change
tag_model="${dataset}_${name_dataset}_${epochs}"

# eval
root="${path_datasets}/${dataset}/${name_dataset}"
output="models/${tag_model}"
model=$output
outdir="preds/$tag_model/$name_dataset"
evaldir="evals/$tag_model/$name_dataset"
subset="test"

python eval.py --model ${model} --targets ${targets} --root ${root} --subset $subset --outdir $outdir --evaldir $evaldir --is-wav 
