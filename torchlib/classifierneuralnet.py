
# STD MODULES 
import os
import math
import shutil
import time
import numpy as np
from tqdm import tqdm

# TORCH MODULE
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# PYTVISION MODULE
from pytvision.neuralnet import NeuralNetAbstract
from pytvision.logger import Logger, AverageFilterMeter, AverageMeter
from pytvision import utils as pytutils
from pytvision import graphic as gph
from pytvision import netlearningrate

# LOCAL MODULE
from . import netmodels as nnmodels
from . import netlosses as nloss


class NeuralNetClassifier(NeuralNetAbstract):
    """
    Convolutional Neural Net 
    """

    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0
        ):
        """
        Initialization
        Args:
            @patchproject (str): path project
            @nameproject (str):  name project
            @no_cuda (bool): system cuda (default is True)
            @parallel (bool)
            @seed (int)
            @print_freq (int)
            @gpu (int)
        """

        super(NeuralNetClassifier, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu  )

 
    def create(self, 
        arch, 
        num_output_channels, 
        num_input_channels,  
        loss, 
        lr, 
        momentum, 
        optimizer, 
        lrsch,          
        pretrained=False,
        topk=(1,),
        ):
        """
        Create
        Args:
            @arch (string): architecture
            @num_output_channels, 
            @num_input_channels,  
            @loss (string):
            @lr (float): learning rate
            @momentum,
            @optimizer (string) : 
            @lrsch (string): scheduler learning rate
            @pretrained (bool)
        """
        super(NeuralNetClassifier, self).create( arch, num_output_channels, num_input_channels, loss, lr, momentum, optimizer, lrsch, pretrained)
        self.accuracy = nloss.Accuracy( topk )
        self.cnf = nloss.ConfusionMeter( self.num_output_channels, normalized=True )
        self.visheatmap = gph.HeatMapVisdom( env_name=self.nameproject )

        # Set the graphic visualization
        self.metrics_name =  [ 'top{}'.format(k) for k in topk ]
        self.logger_train = Logger( 'Trn', ['loss'], self.metrics_name, self.plotter  )
        self.logger_val   = Logger( 'Val', ['loss'], self.metrics_name, self.plotter )
              

      
    def training(self, data_loader, epoch=0):

        self.logger_train.reset()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.train()

        end = time.time()
        for i, sample in enumerate(data_loader):
            
            # measure data loading time
            data_time.update(time.time() - end)
            # get data (image, label)
            inputs, targets = sample['image'], pytutils.argmax(sample['label'])
            batch_size = inputs.size(0)

            if self.cuda:
                targets = targets.cuda( non_blocking=True )
                inputs_var  = Variable(inputs.cuda(),  requires_grad=False)
                targets_var = Variable(targets.cuda(), requires_grad=False)
            else:
                inputs_var  = Variable(inputs,  requires_grad=False)
                targets_var = Variable(targets, requires_grad=False)

            # fit (forward)
            outputs = self.net(inputs_var)

            # measure accuracy and record loss
            loss = self.criterion(outputs, targets_var)            
            pred = self.accuracy(outputs.data, targets )
              
            # optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                      
            # update
            self.logger_train.update(
                {'loss': loss.data[0] },
                dict(zip(self.metrics_name, [pred[p][0] for p in range(len(self.metrics_name))  ])),      
                batch_size,
                )
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:  
                self.logger_train.logger( epoch, epoch + float(i+1)/len(data_loader), i, len(data_loader), batch_time,   )
    

    def evaluate(self, data_loader, epoch=0):
        
        self.logger_val.reset()
        self.cnf.reset()
        batch_time = AverageMeter()
        

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, sample in enumerate(data_loader):
                
                # get data (image, label)
                inputs, targets = sample['image'], pytutils.argmax(sample['label'])
                batch_size = inputs.size(0)

                if self.cuda:
                    targets = targets.cuda( non_blocking=True )
                    inputs_var  = Variable(inputs.cuda(),  requires_grad=False, volatile=True)
                    targets_var = Variable(targets.cuda(), requires_grad=False, volatile=True)
                else:
                    inputs_var  = Variable(inputs,  requires_grad=False, volatile=True)
                    targets_var = Variable(targets, requires_grad=False, volatile=True)
                
                # fit (forward)
                outputs = self.net(inputs_var)

                # measure accuracy and record loss
                loss = self.criterion(outputs, targets_var)            
                pred = self.accuracy(outputs.data, targets )
                self.cnf.add( outputs.argmax(1) , targets_var ) 

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # update
                self.logger_val.update(
                {'loss': loss.data[0] },
                dict(zip(self.metrics_name, [pred[p][0] for p in range(len(self.metrics_name))  ])),      
                batch_size,
                )

                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch, epoch, i,len(data_loader), 
                        batch_time, 
                        bplotter=False,
                        bavg=True, 
                        bsummary=False,
                        )

        #save validation loss
        self.vallosses = self.logger_val.info['loss']['loss'].avg
        acc = self.logger_val.info['metrics']['top1'].avg

        self.logger_val.logger(
            epoch, epoch, i, len(data_loader), 
            batch_time,
            bplotter=True,
            bavg=True, 
            bsummary=True,
            )
        
        print('Confusion Matriz')
        print(self.cnf.value(), flush=True)
        print('\n')
        
        self.visheatmap.show('Confusion Matriz', self.cnf.value())                
        return acc
    
    #def _to_end_epoch(self, epoch, epochs, train_loader, val_loader):
        #print('>> Reset', flush=True )
        #w = 1-self.cnf.value().diagonal()        
        #train_loader.dataset.reset(w)
    

    def test(self, data_loader):
         
        n = len(data_loader)*data_loader.batch_size
        Yhat = np.zeros((n, self.num_output_channels ))
        Y = np.zeros((n,1) )
        k=0

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, sample in enumerate( tqdm(data_loader) ):
                
                # get data (image, label)
                inputs  = sample['image']      
                targets = pytutils.argmax(sample['label']) 
                
                x = inputs.cuda() if self.cuda else inputs    
                x  = Variable(x, requires_grad=False, volatile=True )
                
                # fit (forward)
                yhat = self.net(x)
                yhat = F.softmax(yhat, dim=1)    
                yhat = pytutils.to_np(yhat)
    
                for j in range(yhat.shape[0]):
                        Y[k] = targets[j]
                        Yhat[k,:] = yhat[j] 
                        k+=1 

                #print( 'Test:', i , flush=True )

        Yhat = Yhat[:k,:]
        Y = Y[:k]
                
        return Yhat, Y
    
    def predict(self, data_loader):
         
        n = len(data_loader)*data_loader.batch_size
        Yhat = np.zeros((n, self.num_output_channels ))
        Ids = np.zeros((n,1) )
        k=0

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, (Id, inputs) in enumerate( tqdm(data_loader) ):
                
                # get data (image, label)
                #inputs = sample['image']                
                #Id = sample['id']
                
                x = inputs.cuda() if self.cuda else inputs    
                x  = Variable(x, requires_grad=False, volatile=True )
                
                # fit (forward)
                yhat = self.net(x)
                yhat = F.softmax(yhat, dim=1)    
                yhat = pytutils.to_np(yhat)
    
                for j in range(yhat.shape[0]):
                        Yhat[k,:] = yhat[j]
                        Ids[k] = Id[j]  
                        k+=1 

        Yhat = Yhat[:k,:]
        Ids = Ids[:k]
                
        return Ids, Yhat
      
    def __call__(self, image):        
        
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image    
            x  = Variable(x, requires_grad=False, volatile=True )
            msoft = nn.Softmax()
            yhat = msoft(self.net(x))
            yhat = pytutils.to_np(yhat)

        return yhat


    def representation(self, data_loader):
        """"
        Representation
            -data_loader: simple data loader for image
        """
                
        # switch to evaluate mode
        self.net.eval()

        n = len(data_loader)*data_loader.batch_size
        k=0

        # embebed features 
        embX = np.zeros([n,self.net.dim])
        embY = np.zeros([n,1])

        batch_time = AverageMeter()
        end = time.time()
        for i, sample in enumerate(data_loader):
                        
            # get data (image, label)
            inputs, targets = sample['image'], pytutils.argmax(sample['label'])
            inputs_var = pytutils.to_var(inputs, self.cuda, False, True )

            # representation
            emb = self.net.representation(inputs_var)
            emb = pytutils.to_np(emb)
            for j in range(emb.shape[0]):
                embX[k,:] = emb[j,:]
                embY[k] = targets[j]
                k+=1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Representation: |{:06d}/{:06d}||{batch_time.val:.3f} ({batch_time.avg:.3f})|'.format(i,len(data_loader), batch_time=batch_time) )


        embX = embX[:k,:]
        embY = embY[:k]

        return embX, embY
    
    def _create_model(self, arch, num_output_channels, num_input_channels, pretrained):
        """
        Create model
            @arch (string): select architecture
            @num_classes (int)
            @num_channels (int)
            @pretrained (bool)
        """    

        self.net = None
        self.size_input = 0     
        
        kw = {'num_classes': num_output_channels, 'num_channels': num_input_channels, 'pretrained': pretrained}
        self.net = nnmodels.__dict__[arch](**kw)
        
        self.s_arch = arch
        self.size_input = self.net.size_input        
        self.num_output_channels = num_output_channels
        self.num_input_channels = num_input_channels

        if self.cuda == True:
            self.net.cuda()
        if self.parallel == True and self.cuda == True:
            self.net = nn.DataParallel(self.net, device_ids= range( torch.cuda.device_count() ))

    def _create_loss(self, loss):

        # create loss
        if loss == 'cross':
            self.criterion = nn.CrossEntropyLoss().cuda()
        elif loss == 'mse':
            self.criterion = nn.MSELoss(size_average=True).cuda()
        elif loss == 'l1':
            self.criterion = nn.L1Loss(size_average=True).cuda()
        else:
            assert(False)

        self.s_loss = loss

