#!/bin/bash

# choose the cuda id
export CUDA_VISIBLE_DEVICES=0
echo CUDA ID: $CUDA_VISIBLE_DEVICES

# set args
dataset="musdbn"
name_dataset="musdb18_wav"
#name_model="OpenUnmixMixup"
name_model="OpenUnmixHidden"
epochs="1"  # 1001: alpha=1, 1002: alpha=2, 1003: alpha=2 and .25<lam<.75, 1004: full dataset, 1005(stoped): patience:500, 1006: 7secHidden, 1007: 7secMixup, 1008: new loader, input_mixup, 1009(stoped): layer01mixup, 1010(hope): full dataset, 1011: full, 1012: 256 sample, batch_size32, 1014: checking lam, 1015: checking lam, new dataset, 1019: full, 1020: full, lr.0001, 1021: unif[.9, 1.1] debug, 1021: check mean std, 1022: debug, energy, hidden, 1023: last layer min, 1024: new train hidden, 1025: new train, mixup, 1026: new dataloader, 1027: energy and min full 

targets="vocals drums bass other"
samples_per_track="64"

# don't change
tag_model="${dataset}_${name_dataset}_${name_model}_$epochs"

# train
root="/home/russell/russellizadi/datasets/${dataset}/${name_dataset}"
output="models/${tag_model}"

for target in $targets; do
	python train.py --dataset ${dataset} --root ${root} --name-model ${name_model} --epochs $epochs --output $output --target $target --samples-per-track ${samples_per_track} --is-wav
done

# eval
model=$output
outdir="preds/$tag_model/$name_dataset"
evaldir="evals/$tag_model/$name_dataset"
subset="test"
targets=$targets

python eval.py --model ${model} --targets ${targets} --root ${root} --subset $subset --outdir $outdir --evaldir $evaldir --is-wav 
