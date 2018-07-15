import os
import numpy as np
import pandas as pd

from argparse import ArgumentParser
import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn

from pytvision.transforms import transforms as mtrans
from pytvision import visualization as view

from torchlib.datasets  import factory
from torchlib.datasets  import gdata
from torchlib.neuralnet import NeuralNet
from torchlib.multiclassification import product_ruler, sum_ruler, max_ruler, min_ruler, majority_ruler, mean_ruler

from misc import get_transforms_aug, get_transforms_det, get_transforms_hflip, get_transforms_gray
from sklearn import metrics


def arg_parser():
    """Arg parser"""    
    parser = ArgumentParser()
    parser.add_argument('data', metavar='DIR', 
                        help='path to dataset')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('-g', '--gpu', default=0, type=int, metavar='N',
                        help='divice number (default: 0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', 
                        help='mini-batch size (default: 256)')
    parser.add_argument('--project', default='./out/netruns', type=str, metavar='PATH',
                        help='path to project (default: ./runs)')
    parser.add_argument('--out', default='./out', type=str, metavar='PATH',
                        help='path to output')
    parser.add_argument('--name', default='exp', type=str,
                        help='name of experiment')
    parser.add_argument('--project', default='./runs', type=str, metavar='PATH',
                        help='path to project (default: ./runs)')
    parser.add_argument('--path-model', default='model_best.pth.tar', type=str, metavar='NAME',
                    help='pathname model')
    parser.add_argument('--channels', default=1, type=int, metavar='N',
                        help='input channel (default: 1)')
    return parser



def tta_predict(network, path_model, args):
    
    # load model
    if network.load( path_model ) is not True:
        return 0

    tta_preprocess = [ 
        get_transforms_det(network.size_input), 
        get_transforms_hflip(network.size_input), 
        get_transforms_gray(network.size_input),
        get_transforms_aug(network.size_input),
        get_transforms_aug(network.size_input), 
        ]
    
    dataloaders = []
    for transform in tta_preprocess:    
        # test dataset
        data = gdata.Dataset(
        data=factory.FactoryDataset.factory(
            pathname=args.data, 
            name=args.name_dataset, 
            subset=factory.validation, 
            download=True ),
            num_channels=network.num_input_channels,
            transform=transform,
        )
        
        dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers )
        dataloaders.append(dataloader)

    # print neural net class
    print('NeuralNet: {}'.format(datetime.datetime.now()), flush=True  )
    #print(network)
    
    pathnameout = args.out
    print('dir: {}'.format(pathnameout))
    files = [ f for f in sorted(os.listdir(pathnameout)) if f.split('.')[-1] == 'csv' ]; 
    
    for i,data in enumerate(dataloaders):
        Yhat, Y = network.test( data )
        df = pd.DataFrame( np.concatenate((Y, Yhat), axis=1) )
        df.to_csv( os.path.join(pathnameout , 'val_dp{}.csv'.format(i + len(files) ) ), index=False, encoding='utf-8')     
        
    print('DONE !!!')
    

def evauate( pathnameout ):
    
    files = [ f for f in sorted(os.listdir(pathnameout)) if f.split('.')[-1] == 'csv' ]; 
    print(files)

    l = len(files)
    dp =[]; ys=[]
    for f in files:  
        mdata = pd.read_csv( os.path.join(pathnameout , f )  )
        dpdata = mdata.as_matrix()
        ys.append(dpdata[:,0])    
        dp.append(dpdata[:,1:])

    dp = np.array(dp).transpose((1,2,0))
    ys = np.array(ys)

    assert( not (ys[0,:]-ys[:1,:]).sum() )
    print(dp.shape)
    
    y = ys[0,:]

    # individual result
    print('\nIndividual: ')
    p = np.argmax(dp,axis=1)
    for i in range(p.shape[1]):
        pred = p[:,i]
        acc = (pred==y).astype(float).sum()/len(y)
        print('model_{}:\t{}'.format(i, acc) )

    # multiclasification result
    print('\nMulticlasification: ')
    func = [product_ruler, sum_ruler, max_ruler, min_ruler, majority_ruler, mean_ruler]
    for f in func:
        p = f(dp)
        pred = np.argmax(p, axis=1)
        acc = (pred==y).astype(float).sum()/len(y)
        print('{}:\t{}'.format(f.__name__, acc))


def main():
    
    # parameters
    parser = arg_parser();
    args = parser.parse_args();

    # neuralnet
    network = NeuralNet(
        patchproject=args.project,
        nameproject=args.name,
        no_cuda=args.no_cuda,
        seed=args.seed,
        gpu=args.gpu
        )

    cudnn.benchmark = True    
    tta_predict(network, args.path_model, args)
    evaluate( args.out )
    
    
    
    
    
if __name__ == '__main__':
    main()