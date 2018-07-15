

import os
import numpy as np
from torchvision import datasets 

from . import imaterialist



def create_folder(pathname, name):    
    # create path name dir        
    pathname = os.path.join(pathname, name )
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    return pathname

class FactoryDataset(object):
    
    training = 'train'
    validation = 'val'
    test = 'test'

    mnist='mnist'
    fashion='fashion'
    emnist='emnist'
    cifar10='cifar10'
    cifar100='cifar100'
    stl10='stl10'
    svhn='svhn'

    imaterialist='imaterialist'
    
    
    @classmethod
    def _checksubset(self, subset): 
        return subset=='train' or subset=='val' or subset=='test'

    @classmethod
    def factory(self,         
        pathname,
        name,
        subset='train',
        download=False,
        ):
        """Factory dataset
        """

        assert( self._checksubset(subset) )
        pathname = os.path.expanduser(pathname)
        

        # pythorch vision dataset soported

        if name == 'mnist':   
            btrain=(subset=='train')
            pathname = create_folder(pathname, name)
            data = datasets.MNIST( pathname, train=btrain, download=download)       
            data.labels = np.array( data.targets )        

        elif name == 'fashion':
            btrain=(subset=='train')
            pathname = create_folder(pathname, name)
            data = datasets.FashionMNIST(pathname, train=btrain, download=download)
            data.labels = np.array( data.targets )

        elif name == 'emnist':            
            btrain=(subset=='train')
            pathname = create_folder(pathname, name)
            data = datasets.EMNIST(pathname, split='byclass', train=btrain, download=download)
            data.labels = np.array( data.targets )  

        elif name == 'cifar10':     
            btrain=(subset=='train')  
            pathname = create_folder(pathname, name)     
            data = datasets.CIFAR10(pathname, train=btrain, download=download)
            data.labels = np.array( data.targets )  

        elif name == 'cifar100':  
            btrain=(subset=='train')  
            pathname = create_folder(pathname, name)          
            data = datasets.CIFAR100(pathname, train=btrain, download=download)
            data.labels = np.array( data.targets )

        elif name == 'stl10':  
            split= 'train' if (subset=='train')  else 'test'
            pathname = create_folder(pathname, name)          
            data = datasets.STL10(pathname, split=split, download=download)

        elif name == 'svhn':
            split= 'train' if (subset=='train')  else 'test'
            pathname = create_folder(pathname, name)          
            data = datasets.SVHN(pathname, split=split, download=download)
            data.classes = np.unique( data.labels )



        # kaggle dataset
        elif name == 'imaterialist':
            pathname = create_folder(pathname, name)
            data = imaterialist.IMaterialistDatset(pathname, subset, 'jpg')


            
        else: 
            assert(False)

        data.btrain = (subset=='train')
        return data
