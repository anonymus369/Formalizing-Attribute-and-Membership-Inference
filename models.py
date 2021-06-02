#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ConvNNCifar10(nn.Module):
    """Convolutional Neural Net for Classification
    """
    def __init__(self,num_classes):
        super(ConvNNCifar10, self).__init__()
        self.num_classes = num_classes
        
        def blockLinear(in_features,out_features):
            Layers = [nn.Linear(in_features,out_features)]
            Layers.append(nn.BatchNorm1d(out_features))
            Layers.append(nn.ReLU())
            return Layers

        def blockConv2D(in_channels, out_channels, kernel_size, stride, padding):
            Layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)]
            Layers.append(nn.BatchNorm2d(out_channels))
            Layers.append(nn.ReLU())
            return Layers
        
        self.model = nn.Sequential(
        *blockConv2D(3,16,7,1,2),
        *blockConv2D(16,32,6,2,2),
        *blockConv2D(32,64,5,1,2),
        *blockConv2D(64,64,5,2,2),
        Flatten(),
        *blockLinear(4096,64),
        *blockLinear(64,32),
        nn.Linear(32,num_classes)
        )
        
    def forward(self,x):
        return self.model(x)
    
class ConvNNMNIST(nn.Module):
    """Convolutional Neural Net for Classification
    """
    def __init__(self,num_classes):
        super(ConvNNMNIST, self).__init__()
        self.num_classes = num_classes
        
        def blockLinear(in_features,out_features):
            Layers = [nn.Linear(in_features,out_features)]
            Layers.append(nn.BatchNorm1d(out_features))
            Layers.append(nn.ReLU())
            return Layers

        def blockConv2D(in_channels, out_channels, kernel_size, stride, padding):
            Layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)]
            Layers.append(nn.BatchNorm2d(out_channels))
            Layers.append(nn.ReLU())
            return Layers
        
        self.model = nn.Sequential(
        *blockConv2D(1,16,7,1,2),
        *blockConv2D(16,32,6,2,2),
        *blockConv2D(32,64,5,1,2),
        *blockConv2D(64,64,5,2,2),
        Flatten(),
        *blockLinear(3136,64),
        *blockLinear(64,32),
        nn.Linear(32,num_classes)
        )
        
    def forward(self,x):
        return self.model(x)
    
class PenDigNN(nn.Module):
    """Fully-connected Neural Net for Classification
    """
    def __init__(self,input_size):
        super(PenDigNN, self).__init__()
        
        def blockLinear(in_features,out_features):
            Layers = [nn.Linear(in_features,out_features)]
            Layers.append(nn.BatchNorm1d(out_features))
            Layers.append(nn.ReLU())
            return Layers        

        self.layer1 = nn.Sequential(
        *blockLinear(input_size,32)
        )
        
        self.layer2 = nn.Sequential(
        *blockLinear(32,16)
        )
        
        self.layer3 = nn.Sequential(
        *blockLinear(16,16)
        )        
        
        self.layer4 = nn.Linear(16,10)
        
    def forward(self,images,ids,strokes,time):
        images = torch.reshape(images,(-1,64))
        strokes = torch.unsqueeze(strokes,1)
        time = torch.unsqueeze(time,1)
        in1 = torch.cat((images,ids,strokes,time),dim=1)
        out1 = self.layer1(in1)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        return self.layer4(out3)
