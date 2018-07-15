

mkdir $HOME/.datasets/imaterialist
mkdir $HOME/.datasets/imaterialist/train
mkdir $HOME/.datasets/imaterialist/val
mkdir $HOME/.datasets/imaterialist/test

python ../download_imaterialist.py \
~/.kaggle/competitions/imaterialist-challenge-furniture-2018/train.json \
$HOME/.datasets/imaterialist/download/train \

python ../download_imaterialist.py \
~/.kaggle/competitions/imaterialist-challenge-furniture-2018/validation.json \
$HOME/.datasets/imaterialist/download/val

python ../download_imaterialist.py \
~/.kaggle/competitions/imaterialist-challenge-furniture-2018/test.json \
$HOME/.datasets/imaterialist/download/test
