#!/bin/bash

# train and eval
export CUDA_VISIBLE_DEVICES=3
echo CUDA ID: $CUDA_VISIBLE_DEVICES

# set args
dataset="musdb"

name_dataset="musdb18_debug"
#name_dataset="musdb18_debug"
name_model="OpenUnmixHidden"
#name_model="OpenUnmixMixup"
#name_model="OpenUnmix"

#tag_model="umx_101"
#tag_model="musdb_musdb18_wav_OpenUnmixHidden_1023"
tag_model="musdbn_musdb18_debug_OpenUnmixHidden_1026"
#tag_model="must"

targets="vocals"
#targets="vocals drums bass other"
subset="test"
num_cores="1"
no_cuda="0"
is_wav="1"

# don't change
path_dataset="/home/russell/russellizadi/datasets/$dataset/$name_dataset"
path_model="models/$tag_model"
path_pred="preds/$tag_model/$name_dataset"
path_eval="evals/$tag_model/$name_dataset"


if [ $no_cuda -eq 1 -a $is_wav -eq 1 ]; then
	python eval.py --targets $targets --model $path_model --outdir $path_pred --evaldir $path_eval --root $path_dataset --subset $subset --cores $num_cores --no-cude --is-wav
elif [ $no_cuda -eq 1 -a $is_wav -eq 0 ]; then
	python eval.py --targets $targets --model $path_model --outdir $path_pred --evaldir $path_eval --root $path_dataset --subset $subset --cores $num_cores --no-cude
elif [ $no_cuda -eq 0 -a $is_wav -eq 1 ]; then
	python eval.py --targets $targets --model $path_model --outdir $path_pred --evaldir $path_eval --root $path_dataset --subset $subset --cores $num_cores --is-wav
else
	python eval.py --targets $targets --model $path_model --outdir $path_pred --evaldir $path_eval --root $path_dataset --subset $subset --cores $num_cores
fi
