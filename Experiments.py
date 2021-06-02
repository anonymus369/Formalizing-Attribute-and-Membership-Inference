#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
from numpy import random as rd
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from models import ConvNNCifar10, ConvNNMNIST

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def get_n_params(model):
    """Get the number of parameters in 'model'
    """
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def To1hot(in_tensor,num_class):
    """Output a 1 hot encoded version of 'in_tensor'
    """
    if len(in_tensor.shape) == 1:
        in_tensor = torch.unsqueeze(in_tensor,1)
    onehot = Tensor(in_tensor.shape[0], num_class)
    onehot.zero_()
    onehot.scatter_(1, in_tensor, 1)
    return onehot

def TrainingSet(n,X,Y,seed):
    """ Take n samples from (X,Y) uniformly and then split into training and validation set.
    """
    
    lenX = X.shape[0]
    if n>lenX:
        raise ValueError 
    
    # Randomly pick samples from the universe.
    rd.seed(seed)
    index = rd.choice(lenX,n*2//3,replace=False)
    
    trainX = X[index]
    trainY = Y[index]
    
    # The part of the universe not used for training is used as validation set.
    valIndex = np.setdiff1d(range(lenX),index)
    valIndex = rd.choice(valIndex,n//3,replace=False)
    
    valX = X[valIndex]
    valY = Y[valIndex]
    
    return trainX, trainY, valX, valY
    
class TrainingDataMSE(Dataset):
    """Preprocess training data
    """
    def __init__(self,X,Y):
        self.X = Tensor(X)
        indexY = Tensor(Y).to(dtype=torch.long)
        self.Y = To1hot(indexY,10) 
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,index):
        return (self.X[index,:,:,:],self.Y[index])

def TrainNN_MSE(n, trainX, trainY, valX, valY, mode, seed=5, batch_size=50, n_epochs=2500, epsilon=1e-3):
    """ Instantiates and trains a Neural network for classification. 
    'mode' indicates whether the model is designed for Cifar10 or the MNIST/FashionMNIST sets.
    n: total number of samples used for training (traning + validation).
    (trainX,trainY): training set.
    (ValX,ValY): Validation set.
    """    
    lr = 0.005 #Learning rate for the optimizer

    trainingData = TrainingDataMSE(trainX,trainY)
    valX, valY = Tensor(valX), Tensor(valY).to(dtype=torch.long)
    valY = To1hot(valY,10)
    
    dataloader = DataLoader(trainingData,batch_size=batch_size,shuffle=True)
    
    Len = trainX.shape[0]
    valLen = valX.shape[0]
    
    # Instantiating model, loss function and softmax
    if mode == 'Cifar10':
        NN = ConvNNCifar10(10)
    elif mode == 'MNIST':
        NN = ConvNNMNIST(10)
    Loss = nn.MSELoss()
    Soft = nn.Softmax(dim=1)
    
    if cuda:
        NN.cuda()
        Loss.cuda()
        Soft.cuda()
    
    optimizer = torch.optim.Adam(NN.parameters(),lr=lr)
    currLoss = math.inf
      
    for k in range(n_epochs): # Loop through epochs.
        lostList = []
        Acc = 0
        for i,batch in enumerate(dataloader): # Loop through batches.
            optimizer.zero_grad()
            
            example = batch[0]
            target = batch[1]
            loss = Loss(Soft(NN(example)),target) # Compute the loss.
            loss.backward() # Compute the gradient.
            optimizer.step() # Update the weights.       
            
            aux = Tensor.cpu(sum(torch.eq(torch.argmax(NN(example),1),torch.argmax(target,1))))
            Acc = Acc + aux.data.numpy()                                               
            
            lostList.append(loss.item())
        
        # Compute Accuracy over training set.
        Acc = Acc/Len
        
        # Compute Accuracy over validation set.
        
        aux = Tensor.cpu(torch.eq(torch.argmax(NN(valX),1),torch.argmax(valY,1)))
        valAcc = sum(aux.data.numpy())/valLen
        
        prevLoss = currLoss
        currLoss = np.mean(lostList)

        if (abs(prevLoss-currLoss) < epsilon): # Early stop criteria.
            break

    print('Loss : %f, Accuracy: %f, Validation Accuracy: %f Iteration: %d' % (currLoss, Acc, valAcc, k+1))
    return NN

def Experiment2(n,X,Y,testX,testY,seed,mode=None,precision=10000):
    """Draw a training set of size 'n' randomly, train a model and perform the likelihood attack 
    'precision' number of times on the trained model.
    The generalization error is computed empirically using training set (X,Y) and test set (testX,testY).
    mode: Indicates the model architecture to be initialized and trained.
    """
    if mode is None:
        raise ValueError("mode must be 'Cifar10' or 'MNIST'")
        
    trainX, trainY, valX, valY = TrainingSet(n,X,Y,seed)
    
    NN = TrainNN_MSE(n,trainX,trainY,valX,valY,mode,seed=seed,n_epochs=150)
    
    trainX = np.concatenate((trainX,valX))
    trainY = np.concatenate((trainY,valY))
    
    with torch.no_grad():
        trainXtensor, testXtensor = Tensor(trainX), Tensor(testX)      
        trainYtensor, testYtensor = Tensor(trainY).to(torch.long), Tensor(testY).to(torch.long)
        Loss = nn.MSELoss(reduction='none')
        Soft = nn.Softmax(dim=1)
        
        if cuda:
            Loss.cuda()
            Soft.cuda()
        
        # Likelihood Attack 
        likelihoodTrain = torch.max(Soft(NN(trainXtensor)),1)[0]
        likelihoodTrain = Tensor.cpu(likelihoodTrain)
        likelihoodTrain = likelihoodTrain.data.numpy()
        likelihoodTest = torch.max(Soft(NN(testXtensor)),1)[0]
        likelihoodTest = Tensor.cpu(likelihoodTest)
        likelihoodTest = likelihoodTest.data.numpy()
        
        threshold = .8
        Suc = 0
        for _ in range(precision): # Repeating the attack described in Algorithm 2, "precision" times.
            T = rd.randint(2)
            if T:
                j = rd.randint(len(likelihoodTrain))
                S = likelihoodTrain[j]
            else:
                j = rd.randint(len(likelihoodTest))
                S = likelihoodTest[j] 
            Suc = Suc + int(int(S>threshold)==T)
        Suc = Suc/precision
        
        # Compute Generalization gap.
        
        trainY_onehot = To1hot(trainYtensor,10)
        trainErr = Tensor.cpu(torch.sum(Loss(Soft(NN(trainXtensor)),trainY_onehot),1))
        trainErr = trainErr.data.numpy()
        
        testY_onehot = To1hot(testYtensor,10)
        testErr = Tensor.cpu(torch.sum(Loss(Soft(NN(testXtensor)),testY_onehot),1))
        testErr = testErr.data.numpy()
               
        genErr = abs(np.mean(trainErr)-np.mean(testErr))
        
        # Compute Accuracy on the Test Set
        
        Acc = Tensor.cpu(sum(torch.eq(torch.argmax(NN(testXtensor),1),torch.argmax(testY_onehot,1))))
        Acc = Acc.data.numpy()
        Acc = Acc/testXtensor.shape[0]
        
        return genErr, Suc, Acc
    
