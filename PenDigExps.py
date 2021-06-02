#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import math
import numpy as np
from numpy import random as rd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from models import PenDigNN
from torch.autograd import grad

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def DataArange(myData, n, m, seed):
    """ Takes the data from 'myData' and organizes it into tuples containing: 'images', 'ids', 'strokes', 'time' and 'labels'. The data is split into Training, Validation and Test sets according to 'n' and 'm', as especified by 'DataSplit'.
    """
    images = []
    ids = []
    strokes = []
    time = []
    labels = []

    for i in range(len(myData)):
        images.append([myData[i][0],myData[i][1]])
        ids.append(myData[i][2])
        strokes.append(myData[i][3])
        time.append(myData[i][4])
        labels.append(myData[i][5])

    images = np.asarray(images)
    ids = np.asarray(ids)
    strokes = np.asarray(strokes)
    time = np.asarray(time)
    labels = np.asarray(labels)
    
    return DataSplit(n,m,images,ids,strokes,time,labels,seed)  

def DataSplit(n,m,images,ids,strokes,time,labels,seed):
    """ Shuffles and splits the data into Training, Validation and Test sets. The training set will have size 'n'. The validation set will have size 'm-n', and the rest of the data will be put into the Test set.
    PenDigits data is presented as a tuple containing: 'images', 'ids', 'strokes', 'time' and 'labels'.
    """
    rd.seed(seed)

    indexes = np.arange(images.shape[0])
    rd.shuffle(indexes)

    images = images[indexes]
    ids = ids[indexes]
    strokes = strokes[indexes]
    time = time[indexes]
    labels = labels[indexes]
    
    imagesTra, imagesVal, imagesTes = images[:n,:,:], images[n:m,:,:], images[m:,:,:]
    idsTra, idsVal, idsTes = ids[:n,:], ids[n:m,:], ids[m:,:]
    strokesTra, strokesVal, strokesTes = strokes[:n], strokes[n:m], strokes[m:]
    timeTra, timeVal, timeTes = time[:n], time[n:m], time[m:]
    labelsTra, labelsVal, labelsTes = labels[:n,:], labels[n:m,:], labels[m:,:]
    return((imagesTra,idsTra,strokesTra,timeTra,labelsTra),
           (imagesVal,idsVal,strokesVal,timeVal,labelsVal),
           (imagesTes,idsTes,strokesTes,timeTes,labelsTes))

class TrainingData(Dataset):
    """Preprocess training data
    """
    def __init__(self,images,ids,strokes,time,labels):
        self.images = Tensor(images)
        self.ids = Tensor(ids)
        self.strokes = Tensor(strokes)
        self.time = Tensor(time)
        self.labels = Tensor(labels)
        
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self,index):
        return (self.images[index,:,:],self.ids[index,:],
                self.strokes[index],self.time[index],self.labels[index,:])

def TrainNN_MSE(Tra, Val, seed=5, init=None, batch_size=50, n_epochs=2500, epsilon=1e-3):
    """ Trains a model on Traning set 'Tra', with validation set 'Val'.
    The model is trained for 'n_epochs' epochs, with an early stop criteria controled by 'epsilon'.
    init: The model can initialized with a specific set of parameters.
    """
   
    lr = 0.005 # learning rate for the optimizer

    trainingData = TrainingData(*Tra)
    imagTra, idsTra, stTra, timeTra, labelsTra = Tra[0], Tra[1], Tra[2], Tra[3], Tra[4]
    imagTra, idsTra, stTra, timeTra, labelsTra = Tensor(imagTra), Tensor(idsTra), Tensor(stTra), Tensor(timeTra), Tensor(labelsTra)
    inputsTra = (imagTra,idsTra,stTra,timeTra)
    
    imagVal, idsVal, stVal, timeVal, labelsVal = Val[0], Val[1], Val[2], Val[3], Val[4]
    imagVal, idsVal, stVal, timeVal, labelsVal = Tensor(imagVal), Tensor(idsVal), Tensor(stVal), Tensor(timeVal), Tensor(labelsVal)
    inputsVal = (imagVal,idsVal,stVal,timeVal)
    
    dataloader = DataLoader(trainingData,batch_size=batch_size,shuffle=True)
    
    Len = Tra[0].shape[0]
    valLen = imagVal.shape[0]
    
    input_size = idsTra.shape[1] +66
    NN = PenDigNN(input_size)
    if init is not None:
        NN.load_state_dict(init)
    NN.train()
    
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
        for _,batch in enumerate(dataloader): # Loop through batches.
            optimizer.zero_grad()
            
            example = (batch[0],batch[1],batch[2],batch[3])
            target = batch[4]
            loss = Loss(Soft(NN(*example)),target) # Compute the loss.
            loss.backward() # Compute the gradient.
            optimizer.step() # Update the weights.
            
            aux = Tensor.cpu(torch.eq(torch.argmax(NN(*example),1),torch.argmax(target,1)))
            Acc = Acc + sum(aux.data.numpy())                                               
            
            lostList.append(loss.item())
        
        # Compute Accuracy over training set.
        Acc = Acc/Len
        
        # Compute Accuracy over validation set.
        
        aux = Tensor.cpu(torch.eq(torch.argmax(NN(*inputsVal),1),torch.argmax(labelsVal,1)))
        valAcc = sum(aux.data.numpy())/valLen
        
        prevLoss = currLoss
        currLoss = np.mean(lostList)

        if (abs(prevLoss-currLoss) < epsilon):
            break
            
    with torch.no_grad():
        NN.eval()
        Acc = Tensor.cpu(torch.eq(torch.argmax(NN(*inputsTra),1),torch.argmax(labelsTra,1)))
        Acc = sum(Acc.data.numpy())/Len
        aux = Tensor.cpu(torch.eq(torch.argmax(NN(*inputsVal),1),torch.argmax(labelsVal,1)))
        valAcc = sum(aux.data.numpy())/valLen
        
        print('Loss : %f, Accuracy: %f, Validation Accuracy: %f Iteration: %d' % (currLoss, Acc, valAcc, k+1))
    return NN

def LikelihoodAttack(n,Tra,Tes,seed,precision=10000, epsilon=1e-4):
    """ Trains a model and performs the likelihood attack on it. Computes the generalization gap and accuracy of the model.
    n: Size of the training set.
    Tra: Pool of samples from which the training set will be selected.
    Tes: Test set used to compute the accuracy of the model and the generalization gap.
    precision: Number of times the inference attacks will be performed.
    """
    
    rd.seed(seed)
        
    Tra_, Val_, _ = DataSplit(n-10,n,*Tra,seed)
    
    NN = TrainNN_MSE(Tra_, Val_, epsilon=epsilon, n_epochs=150) # Training target model
    
    imagTra, idsTra, stTra, timeTra, labelsTra = Tra_[0], Tra_[1], Tra_[2], Tra_[3], Tra_[4]
    imagTra, idsTra, stTra, timeTra, labelsTra = Tensor(imagTra), Tensor(idsTra), Tensor(stTra), Tensor(timeTra), Tensor(labelsTra)
    inputsTra = (imagTra,idsTra,stTra,timeTra)
    
    imagTes, idsTes, stTes, timeTes, labelsTes = Tes[0], Tes[1], Tes[2], Tes[3], Tes[4]
    imagTes, idsTes, stTes, timeTes, labelsTes = Tensor(imagTes), Tensor(idsTes), Tensor(stTes), Tensor(timeTes), Tensor(labelsTes)
    inputsTes = (imagTes,idsTes,stTes,timeTes)
    with torch.no_grad():
        Loss = nn.MSELoss(reduction='none')
        Soft = nn.Softmax(1)
        
        if cuda:
            Loss.cuda()
            Soft.cuda()
        
        #Likelihood Attack 
        likelihoodTrain = torch.max(Soft(NN(*inputsTra)),1)[0]
        likelihoodTrain = Tensor.cpu(likelihoodTrain)
        likelihoodTrain = likelihoodTrain.data.numpy()
        likelihoodTest = torch.max(Soft(NN(*inputsTes)),1)[0]
        likelihoodTest = Tensor.cpu(likelihoodTest)
        likelihoodTest = likelihoodTest.data.numpy()
        
        threshold = .8
        Suc = 0
        for _ in range(precision): #Repeat the likelihood attack 'precision' times, as especified by Algorithm 2 in the paper.
            T = rd.randint(2)
            if T:
                j = rd.randint(len(likelihoodTrain))
                S = likelihoodTrain[j]
            else:
                j = rd.randint(len(likelihoodTest))
                S = likelihoodTest[j] 
            Suc = Suc + int(int(S>threshold)==T)
        Suc = Suc/precision
               
        trainErr = Tensor.cpu(torch.sum(Loss(Soft(NN(*inputsTra)),labelsTra),1))
        trainErr = trainErr.data.numpy()

        testErr = Tensor.cpu(torch.sum(Loss(Soft(NN(*inputsTes)),labelsTes),1))
        testErr = testErr.data.numpy()
        
        # Generalization gap
        genErr = abs(np.mean(trainErr)-np.mean(testErr))
        
        # Accuracy on the Test Set
        
        Acc = Tensor.cpu(sum(torch.eq(torch.argmax(NN(*inputsTes),1),torch.argmax(labelsTes,1))))
        Acc = Acc.data.numpy()
        Acc = Acc/labelsTes.shape[0]
        
        return genErr, Suc, Acc
    
def AttributeInference(n, Tra, seed, precision=1000, epsilon=1e-4):
    """ Trains a model and performs different attribute inference strategies, computing the success rate for each.
    n: Size of the training set.
    Tra: Pool of samples from which the training set will be selected.
    precision: Number of times the inference attacks will be performed.
    """
    
    rd.seed(seed)
    
    precision = precision if (precision < n-10) else n-10
    
    Tra_, Val_, _ = DataSplit(n-10,n,*Tra,seed)
    
    NN = TrainNN_MSE(Tra_, Val_, epsilon=epsilon, n_epochs=150) #Training the target model.
    NN.eval()
    
    imagTra, idsTra, stTra, timeTra, labelsTra = Tra_[0], Tra_[1], Tra_[2], Tra_[3], Tra_[4]
    imagTra, idsTra, stTra, timeTra, labelsTra = Tensor(imagTra), Tensor(idsTra), Tensor(stTra), Tensor(timeTra), Tensor(labelsTra)
    inputsTra = (imagTra,idsTra,stTra,timeTra)
    
    idN = idsTra.shape[1]
    candidateIds = torch.eye(idN) # The attacker will try every possible value of the sensitive attribute, and choose the best according to some criteria.
    
    Loss = nn.MSELoss(reduction='none')
    Soft = nn.Softmax(1)

    if cuda:
        Loss.cuda()
        Soft.cuda()

    SucA = 0
    SucB = 0
    SucC = 0
    SucD = 0

    indx = rd.choice(n-10,size=precision,replace=False)
    for i in indx: # Loop over samples in the training set. The attacks will be performed 'precision' times.
        
        # Creating a batch from a single sample, where all the attributes are repeated except the sensitive atribute, which varies from 0 to 43.
        imag = imagTra[i,:,:]
        imag = imag.expand(idN,-1,-1)
        st = stTra[i]
        st = st.expand(idN)
        time = timeTra[i]
        time = time.expand(idN)
        labels = labelsTra[i,:]
        labels = labels.expand(idN,-1)

        inputs = (imag,candidateIds,st,time)

        id_ = torch.argmax(idsTra[i,:])

        # Criteria A: Confidence

        tA = torch.argmax(torch.max(NN(*inputs),dim=1)[0])
        if tA == id_:
            SucA += 1

        # Criteria B: Accuracy

        loss = torch.mean(Loss(Soft(NN(*inputs)),labels),1)
        tB = torch.argmin(loss)
        if tB == id_:
            SucB += 1

        # Criteria C: Loss

        pred = torch.argmax(NN(*inputs),dim=1)
        mask = (pred != torch.argmax(labels,dim=1))
        logits = NN(*inputs)
        logits[mask,:] = torch.zeros(labels.shape[1])

        tC = torch.argmax(torch.max(logits,dim=1)[0])
        max_ = torch.max(logits)
        if (max_ >0) and (tC == id_):
            SucC += 1

        # Criteria D: Gradient norm

        gradNormList = []

        for j in range(loss.shape[0]):
            NNgrad = grad(loss[j],NN.parameters(),create_graph=True)
            NNGnorm = torch.sqrt(sum([grd.norm()**2 for grd in NNgrad]))
            NNGnorm
            NNGnorm = NNGnorm.detach()
            NNGnorm = Tensor.cpu(NNGnorm)
            gradNormList.append(NNGnorm.data.numpy())
        
        tD = np.argmin(gradNormList)
        if tD == id_:
            SucD += 1

    SucA = SucA/precision
    SucB = SucB/precision
    SucC = SucC/precision
    SucD = SucD/precision
        
    return SucA, SucB, SucC, SucD
