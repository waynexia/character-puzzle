import torch.nn as nn
import torch

import data as Data
import model as Model

device = 'cpu'

def train_iter(data,model,criterion,optimizer):
    [dataset_X,dataset_gd,dataset_opts],datset_size = data.get_train_data()
    loss = 0
    for i in range(datset_size):
        X = torch.tensor(dataset_X[i]).to(device)
        gd = dataset_gd[i]
        opts = dataset_opts[i]
        for opt in opts:
            #print(opts,gd)
            y = torch.LongTensor([int(opt == gd),]).to(device)

            opt = torch.tensor(opt).to(device)
            output = model(X,opt)
            
            #optim
            optimizer.zero_grad()
            #print(y.shape)
            #output = torch.tensor(output[0],1-output[0])
            print(output,y)
            loss = criterion(output,y)
            loss.backward()
            optimizer.step()
            print(loss)
            #exit()
    return loss

if __name__ == "__main__":
    data = Data.Data(n_for_1 = 2)
    model = Model.Encoder(voc_size = data.get_voc_size(), hidden_size = 100,n_layers = 1,dropout = 0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.001)

    print(train_iter(data,model,criterion,optimizer))

    
