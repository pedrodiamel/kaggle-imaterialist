#!/bin/bash


# parameters
DATA=$HOME/.datasets/imaterialist/test
PROJECT='../out/netruns'
BATCHSIZE=40
EXP_NAME='exp_net_baseline_resnet50_imaterialist_001'
WORKERS=40
PATHMODEL=$PROJECT/$EXP_NAME/'chk000030.pth.tar' #model_best.pth.tar
GPU=0
NUMCHANNELS=3


python ../submission.py \
$DATA \
--project=$PROJECT \
--name=$EXP_NAME \
--batch-size=$BATCHSIZE \
--workers=$WORKERS \
--gpu=$GPU \
--path-model=$PATHMODEL \
--channels=$NUMCHANNELS \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME"test.log" \
