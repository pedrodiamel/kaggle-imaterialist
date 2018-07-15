#!/bin/bash

# parameters
DATA=$HOME/.datasets
NAMEDATASET='ferp'
PROJECT='../out/netruns'
EPOCHS=60
BATCHSIZE=60
LEARNING_RATE=0.0001
MOMENTUM=0.5
PRINT_FREQ=100
WORKERS=60
RESUME='chk000000xx.pth.tar'
GPU=0
ARCH='alexnet'
LOSS='cross'
OPT='adam'
SCHEDULER='fixed'
SNAPSHOT=5
NUMCLASS=8
NUMCHANNELS=3
EXP_NAME='exp_baseline_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_001'

rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
rm -rf $PROJECT/$EXP_NAME/
mkdir $PROJECT    
mkdir $PROJECT/$EXP_NAME  


## execute
python ../classification_baseline_train.py \
$DATA \
--project=$PROJECT \
--name=$EXP_NAME \
--epochs=$EPOCHS \
--batch-size=$BATCHSIZE \
--learning-rate=$LEARNING_RATE \
--momentum=$MOMENTUM \
--print-freq=$PRINT_FREQ \
--workers=$WORKERS \
--resume=$RESUME \
--gpu=$GPU \
--loss=$LOSS \
--opt=$OPT \
--snapshot=$SNAPSHOT \
--scheduler=$SCHEDULER \
--arch=$ARCH \
--num-classes=$NUMCLASS \
--name-dataset=$NAMEDATASET \
--channels=$NUMCHANNELS \
--finetuning \
--parallel \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \

