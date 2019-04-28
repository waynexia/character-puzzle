import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score,precision_score,recall_score

import model as Model
import data as Data

def valid(model_path,n_for_1,device,batch_size):


    data = Data.Data(n_for_1)
    voc_size = data.get_voc_size()
    model = Model.Encoder(batch_size = batch_size,voc_size = voc_size, hidden_size = 100, device = device ,n_layers = 1,dropout = 0).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("validing")

    [dataset_X,dataset_opt,dataset_gt],datset_size = data.get_valid_data()

    y_gt = [] #ground truth
    y_pd = [] #prediction

    softmax = torch.nn.Softmax(dim = 1).cuda()
    
    for i in range(0,datset_size - batch_size,batch_size):
        # padding X with `voc_size`
        X = dataset_X[i : i + batch_size]
        lens = torch.tensor([len(X[i]) for i in range(len(X))]).to(device)
        max_len = max([len(X[i]) for i in range(len(X))])
        X = [xiter + [voc_size for _ in range(max_len - len(xiter))] for xiter in X]
        X = torch.tensor(X).to(device)

        gt = dataset_gt[i : i + batch_size]
        gt = torch.LongTensor(gt).to(device)
        opt = dataset_opt[i : i + batch_size]
        opt = torch.LongTensor(opt).to(device)

        output = model(X,opt,lens)

        y_gt += gt.tolist()
        y_pd += [pd[1] > 0.5 for pd in softmax(output).tolist()]

    #print(dataset_gt[-3:])
    #print(dataset_opt[-3:])

    #print(y_gt)
    #print(y_pd)
    f1 = f1_score(y_gt, y_pd)
    print('F1: {}'.format(f1))
    print('Precision: {}'.format(precision_score(y_gt, y_pd)))
    print('Recall: {}'.format(recall_score(y_gt, y_pd)))

    return f1

if __name__ == "__main__":
    valid(model_path = "./model9000", n_for_1 = 2, device = "cuda", batch_size = 50) 