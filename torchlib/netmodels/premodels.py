#----------------------------------------------------------------------------------------------
# PYTORCH ZOO
# https://github.com/pytorch/vision/tree/master/torchvision/models
#
# AlexNet       | https://arxiv.org/abs/1404.5997
# VGG11         | http://www.robots.ox.ac.uk/~vgg/research/very_deep/
# ResNet18      | https://arxiv.org/abs/1512.03385
# InceptionV3   | http://arxiv.org/abs/1512.00567
# DenseNet      | https://arxiv.org/pdf/1608.06993.pdf
#
#----------------------------------------------------------------------------------------------

import torch
#import torchvision
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from torchlib.netmodels import utility as utl
from torchlib.netmodels import alexnet
from torchlib.netmodels import vgg
from torchlib.netmodels import resnet
from torchlib.netmodels import inception

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
}


class AlexNet(nn.Module):
    """
    "One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    def __init__(self, num_classes=8, pretrained=False):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.alexnet = alexnet.AlexNet(num_classes=num_classes)
        self.dim = self.alexnet.dim

        if pretrained:     
            utl.load_state_dict(self.alexnet.state_dict(), model_zoo.load_url(model_urls['alexnet']))
            nn.init.xavier_normal(self.alexnet.classifier[6].weight)
            
    def forward(self, x):
        x = self.alexnet(x)
        return x

    def representation(self, x):
        return self.alexnet.representation(x)


class VGG11(nn.Module):
    """
    http://www.robots.ox.ac.uk/~vgg/research/very_deep/
    """
    def __init__(self, num_classes=8, pretrained=False):
        super(VGG11, self).__init__()
        self.num_classes = num_classes       
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        features = vgg.make_layers(cfg,  batch_norm=False)        
        self.vgg11 = vgg.VGG( features, num_classes=num_classes )
        self.dim = self.vgg11.dim
                
        if pretrained:     
            utl.load_state_dict( self.vgg11.state_dict(), model_zoo.load_url(model_urls['vgg11']))
            nn.init.xavier_normal(self.vgg11.classifier[6].weight)
        
    def forward(self, x):
        x = self.vgg11(x)
        return x

    def representation(self, x):
        return self.vgg11.representation(x)


class ResNet18(nn.Module):
    """
    https://arxiv.org/abs/1512.03385
    """
    def __init__(self, num_classes=8, pretrained=False):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes        
        blocbasic = resnet.BasicBlock
        cfg = [2, 2, 2, 2]
        self.resnet18 = resnet.ResNet(blocbasic, cfg,  num_classes=num_classes)   
        self.dim = self.resnet18.dim

        if pretrained:     
            utl.load_state_dict( self.resnet18.state_dict(), model_zoo.load_url(model_urls['resnet18']))
            nn.init.xavier_normal(self.resnet18.fc.weight)
        
    def forward(self, x):
        #x = torch.cat((x, x, x), dim=1)
        x = self.resnet18(x)
        return x

    def representation(self, x):
        return self.resnet18.representation(x)


class InceptionV3(nn.Module):
    """
    Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>
    """
    def __init__(self, num_classes=8, pretrained=False):
        super(InceptionV3, self).__init__()
        self.num_classes = num_classes   
        self.inception = inception.Inception3(num_classes=num_classes, transform_input=False, aux_logits=False)  
        self.dim = self.inception.dim 
        
        if pretrained:     
            utl.load_state_dict(self.inception.state_dict(), model_zoo.load_url(model_urls['inception_v3_google']))
            nn.init.xavier_normal(self.inception.fc.weight)
        
    def forward(self, x):
        #x = torch.cat((x, x, x), dim=1)
        x = self.inception(x)
        return x

    def representation(self, x):
        return self.inception.representation(x)



class DenseNet(nn.Module):
    """
    Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>
    """
    def __init__(self, num_classes=8, pretrained=False):
        super(DenseNet, self).__init__()
        self.num_classes = num_classes
        self.densenet = torchvision.models.DenseNet(num_classes=num_classes, 
            num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16))  
        self.dim = self.densenet.dim

        if pretrained:     
            utl.load_state_dict( self.densenet.state_dict(), model_zoo.load_url(model_urls['densenet121']))
            nn.init.xavier_normal(self.densenet.classifier.weight)
        
    def forward(self, x):
        x = torch.cat((x, x, x), dim=1)
        x = self.densenet(x)
        return x

    def representation(self, x):
        return self.inception.representation(x)





