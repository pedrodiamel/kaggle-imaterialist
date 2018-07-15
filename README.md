# Kaggle iMaterialist Challenge (Furniture) at FGVC5

Image Classification of Furniture & Home Goods.

![Dataset](/rec/dataset.jpg)

Top8%: [31 place](https://www.kaggle.com/c/imaterialist-challenge-furniture-2018/leaderboard)


### This repository

- Data visualization
- Neural net framework in pytorch


### Training visualize

We now support Visdom for real-time loss visualization during training!

To use Visdom in the browser:

    # First install Python server and client 
    pip install visdom
    # Start the server (probably in a screen or tmux)
    python -m visdom.server -env_path runs/visdom/
    # http://localhost:8097/

![Train](/rec/train.png)

### Installation

    $git clone ...
    $pip install -r installation.txt
    $ln -s [path dataset].datasets ~/


### Reference


- https://github.com/pedrodiamel/siamese-triplet
- https://github.com/Cadene/pretrained-models.pytorch
- https://towardsdatascience.com/instance-embedding-instance-segmentation-without-proposals-31946a7c53e1

visualization notebook

- https://github.com/jupyterlab/jupyterlab
- https://towardsdatascience.com/interactive-visualizations-in-jupyter-notebook-3be02ab2b8cd
- https://towardsdatascience.com/a-very-simple-demo-of-interactive-controls-on-jupyter-notebook-4429cf46aabd
- http://jupyter.org/widgets

c++

- https://github.com/leonardvandriel/caffe2_cpp_tutorial#intro-tutorial
- https://github.com/onnx/tutorials/blob/master/tutorials/PytorchCaffe2MobileSqueezeNet.ipynb
- https://github.com/onnx/tutorials
- https://github.com/longcw/pytorch2caffe
-  

others

- https://github.com/neuropoly 
- https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm

