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
from torchlib.datasets.imaterialist import IMaterialistImageDataset
from torchlib.multiclassification import product_ruler, sum_ruler, max_ruler, min_ruler, majority_ruler, mean_ruler

from misc import get_transforms_aug, get_transforms_det, get_transforms_hflip, get_transforms_gray, tta_preprocess
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


def predict( network, path_model, args ):
    
    # load model
    if network.load( args.path_model ) is not True:
        return 0
    
    data = IMaterialistImageDataset(
        pathname=args.data,
        ext='jpg',
        num_channels=network.num_input_channels,
        transform=get_transforms_det(network.size_input)
        )

    dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers )
    dataloaders.append(dataloader)
    
    # print neural net class
    print('NeuralNet: {}'.format(datetime.datetime.now()), flush=True  )
    #print(network)

    Ids, Yhat = network.predict( data )
    return Id, pred
    
    
    

def tta_predict(network, path_model, tta_preprocess, args):
        
    # load model
    if network.load( args.path_model ) is not True:
        return 0
    
    dataloaders = []
    for transform in tta_preprocess:    
        # test dataset
        data = IMaterialistImageDataset(
            pathname=args.data,
            ext='jpg',
            num_channels=network.num_input_channels,
            transform=transform(network.size_input)
            )

        dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers )
        dataloaders.append(dataloader)

    # print neural net class
    print('NeuralNet: {}'.format(datetime.datetime.now()), flush=True  )
    #print(network)
    
    pathnameout = args.out   
    if not os.path.exists(pathnameout):
        os.makedirs(pathnameout)
    
    print('dir: {}'.format(pathnameout))
    files = [ f for f in sorted(os.listdir(pathnameout)) if f.split('.')[-1] == 'csv' ]; 
    
    for i,data in enumerate(dataloaders):
        Ids, Yhat = network.predict( data )
        df = pd.DataFrame( np.concatenate((Ids, Yhat), axis=1) )
        df.to_csv( os.path.join(pathnameout , 'test_dp{}.csv'.format(i + len(files) ) ), index=False, encoding='utf-8')       
    
    print('DONE!!!')
    

def combination( pathnameout, func, args ):   
    
    files = [ f for f in sorted(os.listdir(pathnameout)) if f.split('.')[-1] == 'csv' ]; 
    print(files)

    l = len(files)
    dp =[]; ids=[]
    for f in files:  
        mdata = pd.read_csv( os.path.join(pathnameout , f )  )
        dpdata = mdata.as_matrix()
        ids.append(dpdata[:,0])    
        dp.append( dpdata[:,1:] )

    dp = np.array(dp).transpose((1,2,0))
    ids = np.array(ids)

    assert( not (ids[0,:]-ids[:1,:]).sum() )
    print(dp.shape)
    
    Id = ids[0,:]
    p = func(dp)
    pred = np.argmax(p, axis=1)
    
    return Id, pred
    
    
def submission( Id, pred, args ): 
    
    submission_filepath = 'submission.csv'
    submission = pd.read_csv('~/.kaggle/competitions/imaterialist-challenge-furniture-2018/sample_submission_randomlabel.csv')
    
    data_test = IMaterialistImageDataset(
            pathname=args.data,
            ext='jpg',
            num_channels=network.num_input_channels,
            )
    
    data_val=factory.FactoryDataset.factory(
            pathname=pathname, 
            name=name_dataset, 
            subset=factory.validation, 
            download=True )

    TIds = np.array([ int(data_test.getId( int(i) )) for i in Id ])
    TPred = np.array( [ int(data_val.classes[c]) for c in pred  ] )
    submission.loc[TIds-1, 'predicted'] = TPred
    


def main():
    
    # parameters
    parser = arg_parser();
    args = parser.parse_args();
    use_tta = True
    
    tta_preprocess = [ 
        get_transforms_det, 
        get_transforms_hflip, 
        get_transforms_gray,
        get_transforms_aug,
        get_transforms_aug, 
        ]

    # neuralnet
    network = NeuralNet(
        patchproject=args.project,
        nameproject=args.name,
        no_cuda=args.no_cuda,
        seed=args.seed,
        gpu=args.gpu
        )

    cudnn.benchmark = True
    
    
    if use_tta:
        tta_predict(network, args.path_model, tta_preprocess, args)
        Id, pred = combination( pathnameout, func, args )
    else:
        Id, pred = predict( network, path_model, args )
        
    
    submission( Id, pred, args )    
    print('DONE!!!')
    
    
if __name__ == '__main__':
    main()