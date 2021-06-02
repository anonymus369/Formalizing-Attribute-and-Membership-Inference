#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np

def ParseData(string,mode):
    longstring = ''
    counter = 0
    sampleList = []
    x = []
    y = []
    longstring = ''
    label = None
    identity = None
    strokes = 0
    time = 0
    section = False

    while (counter+1)<len(string): #Loop through the large String

        if longstring[-8:] == '.COMMENT':
            if label is not None:
                sample = [x,y,identity,strokes,time,label]
                sampleList.append(sample)
            x = []
            y = []
            longstring = ''
            label = None
            identity = None
            strokes = 0
            time = 0
            section = True

        if len(longstring)>0:
            if (longstring[-1] != ' ') and (label is None) and section: #Collecting the label
                shortstring = longstring[-1]
                while shortstring[-1] != ' ':
                    counter += 1
                    shortstring = shortstring + string[counter]
                    longstring = longstring + string[counter]
                label = int(shortstring[:-1])

        if len(longstring)>0:
            if (longstring[-1] != ' ') and (identity is None) and section: #Collecting the id
                shortstring = longstring[-1]      
                while shortstring[-1] != ' ':
                    counter += 1
                    shortstring = shortstring + string[counter]
                    longstring = longstring + string[counter]
                identity = int(shortstring[:-1])
                if mode == 'test':
                    identity = identity + 30
                section = False

        if  longstring[-9:] == '.PEN_DOWN': #Reading coordinates
            strokes += 1 #Counting number of strokes
            counter += 1
            longstring = longstring + string[counter]

            xFlag = True
            while longstring[-7:] != '.PEN_UP': #Loop over coordinates
                shortstring = ''
                written = False
                while (longstring[-1] != ' ') and (longstring[-1] != '\n'): #Loop over x
                    shortstring = shortstring + string[counter]
                    counter += 1
                    longstring = longstring + string[counter]
                    written = True
                if shortstring[-7:] == '.PEN_UP':
                    break
                if written:               
                    if xFlag:
                        time += 1
                        x.append(int(shortstring[:])) #Adding new coordinate to the list of x coordinates
                        written = False
                        xFlag = not(xFlag)
                    else:
                        y.append(int(shortstring[:])) #Adding new coordinate to the list of y coordinates
                        written = False
                        xFlag = not(xFlag)                            
                counter += 1
                longstring = longstring + string[counter]

        counter += 1
        longstring = longstring + string[counter]
    return sampleList

def UpSample(x,y,N):
    n = len(x)
    if n > N:
        return x, y
    indexes = np.ones(n-1)
    if 2*n -1 >= N:
        subset = rd.choice(n-1,2*n-1-N,replace=False)
        indexes[subset] = 0
    new_x = [x[0]]
    new_y = [y[0]]
    for i in range(n-1):
        if indexes[i]:
            new_x.append((x[i]+x[i+1])/2)
            new_y.append((y[i]+y[i+1])/2)
        new_x.append(x[i+1])
        new_y.append(y[i+1])
    if len(new_x) == N:
        return new_x, new_y
    elif len(new_x) < N:
        return UpSample(new_x,new_y,N)

def DownSample(x,y,N):
    n = len(x)
    if n < N:
        return x, y 
    
    if n%2 == 0: #if x is of even length add a point in the middle
        aux_x = []
        aux_y = []
        for i in range(n//2):
            aux_x.append(x[i])
            aux_y.append(y[i])
        aux_x.append((x[n//2]+x[n//2-1])/2)
        aux_y.append((y[n//2]+y[n//2-1])/2)
        for i in range(n//2,n):
            aux_x.append(x[i])
            aux_y.append(y[i])
        x = aux_x[:]
        y = aux_y[:]
        
    n = len(x)    
    indexes = np.zeros(n//2)
    if n-n//2 <= N:
        subset = rd.choice(n//2,N-n+n//2,replace=False)
        indexes[subset] = 1
    new_x = []
    new_y = []
    for i in range(0,n-2,2):
        new_x.append(x[i])
        new_y.append(y[i])
        if indexes[i//2]:
            new_x.append(x[i+1])
            new_y.append(y[i+1])
    new_x.append(x[-1])
    new_y.append(y[-1])
    if len(new_x) == N:
        return new_x, new_y
    elif len(new_x) > N:
        return DownSample(new_x,new_y,N)
    
def Rescale(x,y):
    minx = np.amin(x)
    miny = np.amin(y)
    maxx = np.amax(x)
    maxy = np.amax(y)
    Min = np.amin([minx,miny])
    Max = np.amax([maxx,maxy])
    return (x-Min)/(Max-Min), (y-Min)/(Max-Min)

def To1hot(label,num_class):
    """Output a 1 hot encoded version of 'label'
    """
    onehot = np.zeros(num_class)
    onehot[label] = 1
    return onehot

def PenDigPreProcessing(sampleList):

    arrayList = []

    for i in range(len(sampleList)):
        sample = sampleList[i]
        if len(sample[0])>32:
            new_x, new_y = DownSample(sample[0], sample[1],32)
        elif len(sample[0])<32:
            new_x, new_y = UpSample(sample[0], sample[1],32)
        else:
            new_x, new_y = sample[0], sample[1]
        new_x, new_y = np.array(new_x,dtype=np.single), np.array(new_y,dtype=np.single)
        new_x, new_y = Rescale(new_x, new_y)
        new_id = To1hot(sample[2]-1,44)
        new_time = (sample[4]-minTime)/(maxTime-minTime)
        new_label = To1hot(sample[5],10)
        new_sample = [new_x,new_y,new_id.astype(dtype=np.single),
                      np.array(sample[3],dtype=np.single),np.array(new_time,dtype=np.single),
                      new_label.astype(dtype=np.single)]
        arrayList.append(new_sample)
    return arrayList

