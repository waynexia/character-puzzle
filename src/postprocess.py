import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score,precision_score,recall_score

import model as Model
import data as Data

def valid(model_path,device):
    data = Data.Data(n_for_1 = 2)
    model = Model.Encoder(voc_size = data.get_voc_size(), hidden_size = 100,n_layers = 1,dropout = 0).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("validing")

    [dataset_X,dataset_gd,dataset_opts],datset_size = data.get_valid_data()

    y_gt = [] #ground truth
    y_pd = [] #prediction
    softmax = torch.nn.Softmax(dim = 1).cuda()
    for i in range(datset_size):
        X = torch.tensor(dataset_X[i]).to(device)
        gd = dataset_gd[i]
        opts = dataset_opts[i]
        for opt in opts:
            #print(opts,gd)
            y = torch.LongTensor([int(opt == gd),]).to(device)

            opt = torch.tensor(opt).to(device)
            output = model(X,opt)
            output = softmax(output)

            y_pd.append(output[0][1].item() > 0.5)
            y_gt.append(y[0].item())
    
    f1 = f1_score(y_gt, y_pd)
    print('F1: {}'.format(f1))
    print('Precision: {}'.format(precision_score(y_gt, y_pd)))
    print('Recall: {}'.format(recall_score(y_gt, y_pd)))

    return f1

if __name__ == "__main__":
    valid(model_path = "./model_0.22", device = "cuda")