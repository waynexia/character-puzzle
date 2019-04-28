"""
change log:
add softmax in model
omit sigmoid
opt with X feed into model together
use test set loss
"""

import time

import data as Data
import matplotlib.pyplot as plt
import model as Model
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

device = 'cuda'

def train_iter(data,model,criterion,optimizer,batch_size,voc_size):
    [dataset_X,dataset_opt,dataset_gt],datset_size = data.get_train_data()
    loss = 0
    losses = list()
    for i in range(0,datset_size - batch_size,batch_size):
        #time
        begin_time = time.time()

        # get gd & opt
        gt = dataset_gt[i : i + batch_size]
        opt = dataset_opt[i : i + batch_size]
        gt = torch.LongTensor(gt).to(device)

        # append opt to X
        X = dataset_X[i : i + batch_size]
        [X[index].append(opt[index]) for index in range(batch_size)]

        # padding X with `voc_size`
        lens = torch.tensor([len(X[i]) for i in range(len(X))]).to(device)
        max_len = max([len(X[i]) for i in range(len(X))])
        X = [xiter + [voc_size for _ in range(max_len - len(xiter))] for xiter in X]
        X = torch.tensor(X).to(device)

        output = model(X,lens)
        
        #optim
        optimizer.zero_grad()
        loss = criterion(output,gt)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if i % 500 == 0:
            print(i,"/",datset_size," with time: ",time.time()-begin_time,end = "\r")
    return np.average(losses)

def train(max_epoch,batch_size = 5):
    loss = [1,]
    data = Data.Data(n_for_1 = 2)
    voc_size = data.get_voc_size()
    model = Model.Encoder(batch_size = batch_size,voc_size = voc_size, hidden_size = 100, device = device ,n_layers = 1,dropout = 0).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.001)
    epoch_count = 0
    increase_count = 0
    while True:
        print('epoch :',epoch_count)

        train_iter(data,model,criterion,optimizer,batch_size,voc_size)

        # sample
        loss.append(sample(dataset = data, model = model,batch_size = batch_size,criterion = criterion))
        print(loss[-1]," ")
        torch.save(model.state_dict(),"../ckpt/model" + str(epoch_count))

        # judge whether stop or not
        if loss[-2] < loss[-1]:
            increase_count += 1
            if increase_count > 10 and epoch_count > 3000:
                break
        else:
            increase_count = 0
        
        # increase epoch count
        epoch_count += 1

    plt.plot(loss)
    plt.show()

def sample(dataset,model,batch_size,criterion):
    voc_size = dataset.get_voc_size()
    [dataset_X,dataset_opt,dataset_gt],datset_size = dataset.get_test_data()
    y_gt = [] #ground truth
    y_pd = [] #prediction
    for i in range(0,datset_size - batch_size,batch_size):

        # get gd & opt
        gt = dataset_gt[i : i + batch_size]
        opt = dataset_opt[i : i + batch_size]
        gt = torch.LongTensor(gt).to(device)

        # append opt to X
        X = dataset_X[i : i + batch_size]
        [X[index].append(opt[index]) for index in range(batch_size)]

        # padding X with `voc_size`
        lens = torch.tensor([len(X[i]) for i in range(len(X))]).to(device)
        max_len = max([len(X[i]) for i in range(len(X))])
        X = [xiter + [voc_size for _ in range(max_len - len(xiter))] for xiter in X]
        X = torch.tensor(X).to(device)

        output = model(X,lens)

        y_gt += gt.tolist()
        y_pd += output.tolist()

    y_gt = torch.tensor(y_gt)
    y_pd = torch.tensor(y_pd)
    loss = criterion(y_pd,y_gt)
    return loss.item()

if __name__ == "__main__":
    train(max_epoch=10000,batch_size=50)
