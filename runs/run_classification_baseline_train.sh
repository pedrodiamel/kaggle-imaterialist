#!/bin/bash

# experiment
# name: exp_[methods]_[arq]_[num]

# parameters
DATA=$HOME/.datasets
NAMEDATASET='ferp'
PROJECT='../out/netruns'
EPOCHS=30
BATCHSIZE=2
LEARNING_RATE=0.0001
MOMENTUM=0.5
PRINT_FREQ=20
WORKERS=1
RESUME='checkpointxx.pth.tar'
GPU=0
ARCH='simplenet'
LOSS='cross'
OPT='adam'
SCHEDULER='fixed'
SNAPSHOT=10
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
--no-cuda \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \

