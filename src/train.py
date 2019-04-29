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
    for i in range(0,datset_size - batch_size,batch_size):
        #time
        begin_time = time.time()

        # get gd & opt
        gt = dataset_gt[i : i + batch_size]
        opt = dataset_opt[i : i + batch_size]
        gt = torch.LongTensor(gt).to(device)
        gt.detach()

        time_a = (time.time()-begin_time)

        # append opt to X
        X = dataset_X[i : i + batch_size]
        #[X[index].append(opt[index]) for index in range(batch_size)]
        for index in range(batch_size):
            X[index].append(opt[index])
        
        time_b = (time.time()-begin_time-time_a)

        # padding X with `voc_size`
        lens = torch.tensor([len(X[i]) for i in range(len(X))]).to(device)
        max_len = max([len(X[i]) for i in range(len(X))])
        X = [xiter + [voc_size for _ in range(max_len - len(xiter))] for xiter in X]
        X = torch.tensor(X).to(device)

        time_c = (time.time()-begin_time-time_b)

        output = model(X,lens)
        
        time_d = (time.time()-begin_time-time_c)
        
        #optim
        optimizer.zero_grad()

        time_e = (time.time()-begin_time-time_d)

        loss = criterion(output,gt)
        
        time_f = (time.time()-begin_time-time_e)

        loss.backward(retain_graph=True)
        
        time_g = (time.time()-begin_time-time_f)

        optimizer.step()
        loss.detach()
        
        time_h = (time.time()-begin_time-time_g)

        print("",end = "\r")
        
        if i % 500 == 0:
            print(i,"/",datset_size," with time: ","%.7f" %(time.time()-begin_time),"%.7f"%time_a,"%.7f"%time_b,"%.7f"%time_c,"%.7f"%time_d,"%.7f"%time_e,"%.7f"%time_f,"%.7f"%time_g,"%.7f"%time_h,end = "\r")
    return

def train(max_epoch,batch_size = 5):
    loss = [1,]
    data = Data.Data(n_for_1 = 2)
    voc_size = data.get_voc_size()
    model = Model.Encoder(batch_size = batch_size,voc_size = voc_size, hidden_size = 100, device = device ,n_layers = 1,dropout = 0).to(device)
    epoch_count = 0
    increase_count = 0
    while True:
        print('epoch :',epoch_count)

        # without new declaration (seems) make the whole script runs slower and slower
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),lr = 0.001)

        train_iter(data,model,criterion,optimizer,batch_size,voc_size)

        # sample
        loss.append(sample(dataset = data, model = model,batch_size = batch_size))
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
        torch.cuda.empty_cache()

    plt.plot(loss)
    plt.show()

def sample(dataset,model,batch_size):
    criterion = nn.CrossEntropyLoss()
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
