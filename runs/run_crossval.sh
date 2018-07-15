#!/bin/bash


# parameters
PROJECT='../out/netruns'
DATA=$HOME/.datasets/imaterialist/test

PATHNAMEOUT=$PROJECT/'predicts'
BATCHSIZE=40
EXP_NAME='exp_net_baseline_resnet50_imaterialist_005'
WORKERS=40
MODELNAME=$PROJECT/$EXP_NAME/'model_best.pth.tar'
GPU=0
NUMCHANNELS=3


python ../test.py \
$DATA \
--project=$PROJECT \
--name=$EXP_NAME \
--out=$PATHNAMEOUT \
--batch-size=$BATCHSIZE \
--workers=$WORKERS \
--gpu=$GPU \
--path-model=$MODELNAME \
--channels=$NUMCHANNELS \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME"test.log" \
