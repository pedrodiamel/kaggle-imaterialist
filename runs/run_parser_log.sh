#!/bin/bash

PROJECT='./netruns'
EXP_NAME='exp_net_simplenet_lr0001pl_daum_proc_001'
FILENAMEIN=$PROJECT/$EXP_NAME/$EXP_NAME'.log'
FILENAMEOUT=$PROJECT/$EXP_NAME

python parse_log.py \
$FILENAMEIN \
$FILENAMEOUT 