import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np


import data as Data
import model as Model

device = 'cpu'

def train_iter(data,model,criterion,optimizer):
    [dataset_X,dataset_gd,dataset_opts],datset_size = data.get_train_data()
    loss = 0
    losses = list()
    for i in range(datset_size):
        X = torch.tensor(dataset_X[i]).to(device)
        gd = dataset_gd[i]
        opts = dataset_opts[i]
        for opt in opts:
            #print(opts,gd)
            y = torch.LongTensor([int(opt == gd),]).to(device)

            # test: add a man-made correlation
            X = torch.LongTensor(np.array(np.random.rand(X.shape[0]) * 100 + int(opt == gd) * 100,dtype = np.int32))
        
            #print(X)
            #exit()

            opt = torch.tensor(opt).to(device)
            output = model(X,opt)
            
            #optim
            optimizer.zero_grad()
            #print(y.shape)
            #output = torch.tensor(output[0],1-output[0])
            #print(output,y)
            loss = criterion(output,y)
            loss.backward()
            optimizer.step()
            print(loss.item())
            losses.append(loss.item())
            #exit()
    #plt.plot(losses)
    #plt.show()
    return np.average(losses)

if __name__ == "__main__":
    loss = list()
    for _ in range(100):    
        data = Data.Data(n_for_1 = 2)
        model = Model.Encoder(voc_size = data.get_voc_size(), hidden_size = 100,n_layers = 1,dropout = 0)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),lr = 0.0003)

        loss.append(train_iter(data,model,criterion,optimizer))
        print(_,loss[-1])
    plt.plot(loss)
    plt.show()

    
