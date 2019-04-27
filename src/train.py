import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import time


import data as Data
import model as Model

device = 'cuda'

def train_iter(data,model,criterion,optimizer,batch_size,voc_size):
    [dataset_X,dataset_gd,dataset_opts],datset_size = data.get_train_data()
    loss = 0
    losses = list()
    for i in range(0,datset_size - batch_size,batch_size):
        #time
        begin_time = time.time()

        # padding X with `voc_size`
        X = dataset_X[i : i + batch_size]
        lens = torch.tensor([len(X[i]) for i in range(len(X))]).to(device)
        max_len = max([len(X[i]) for i in range(len(X))])
        X = [xiter + [voc_size for _ in range(max_len - len(xiter))] for xiter in X]

        X = torch.tensor(X).to(device)
        gd = dataset_gd[i : i + batch_size]
        opts = dataset_opts[i : i + batch_size]
        #i += batch_size
        opts = [[opt[0] for opt in opts], [opt[1] for opt in opts]]
        #print(opts)
        for opt in opts:
            y = torch.LongTensor([ opt[_] == gd[_] for _ in range(batch_size)]).to(device)

            opt = torch.tensor(opt).to(device)
            output = model(X,opt,lens)
            
            #optim
            optimizer.zero_grad()
            #print(output.shape,y.shape)
            loss = criterion(output,y)
            loss.backward()
            optimizer.step()
            #print(loss.item())
            losses.append(loss.item())
        print(i,"/",datset_size," with time: ",time.time()-begin_time,end = "\r")
    return np.average(losses)

def train(max_epoch,batch_size = 5):
    loss = list()
    data = Data.Data(n_for_1 = 2)
    voc_size = data.get_voc_size()
    model = Model.Encoder(batch_size = batch_size,voc_size = voc_size, hidden_size = 100, device = device ,n_layers = 1,dropout = 0).to(device)
    for _ in range(max_epoch):
        print('epoch :',_)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),lr = 0.0003)

        loss.append(train_iter(data,model,criterion,optimizer,batch_size,voc_size))
        print(_,loss[-1])
    torch.save(model.state_dict(),"./model" + str(time.time))
    plt.plot(loss)
    plt.show()

if __name__ == "__main__":
    train(max_epoch=1000,batch_size=25)

    
