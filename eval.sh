#!/bin/bash

# choose the cuda id
export CUDA_VISIBLE_DEVICES=2
echo CUDA ID: $CUDA_VISIBLE_DEVICES

# set args
path_datasets="/home/russell/russellizadi/datasets"
dataset="musdb"
name_dataset="musdb18_wav"
targets="vocals"
#targets="vocals drums bass other"
is_wav="--is-wav"

# don't change
tag_model="musdb_musdb18_wav_1002"
#tag_model="umx_101"
# eval
root="${path_datasets}/${dataset}/${name_dataset}"
output="models/${tag_model}"
model=$output
outdir="preds/$tag_model/$name_dataset"
evaldir="evals/$tag_model/$name_dataset"
subset="test"

python eval.py --model ${model} --targets ${targets} --root ${root} --subset $subset --outdir $outdir --evaldir $evaldir --is-wav 
