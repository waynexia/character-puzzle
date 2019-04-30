import numpy as np
import matplotlib.pyplot as plt
import torch

import model as Model
import data as Data

def valid(model_path,n_for_1,device,batch_size):


    dataset = Data.Data(n_for_1)
    voc_size = dataset.get_voc_size() -1
    model = Model.Encoder(batch_size = batch_size,voc_size = voc_size, hidden_size = 100, device = device ,n_layers = 1,dropout = 0).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("validing")
    total_cnt = 0
    correct_cnt = 0



    [dataset_X,dataset_gt,dataset_opt],dataset_size = dataset.get_valid_data()
    for batch_iter_cnt in range(0,dataset_size - batch_size,batch_size):
        # prediction
        # shape is `n_for_1` * `batch_size`
        y_pd = []

        # get gd & opt
        gt = dataset_gt[batch_iter_cnt : batch_iter_cnt + batch_size]
        opt = dataset_opt[batch_iter_cnt : batch_iter_cnt + batch_size]
        gt = torch.LongTensor(gt).to(device)

        # get X
        X = dataset_X[batch_iter_cnt : batch_iter_cnt + batch_size]
        # search for each opt available
        for opt_index in range(n_for_1):
            # append `this opt` to X
            X_with_opt = [X[index]+ [opt[index][opt_index],] for index in range(batch_size)]

            # padding X with `voc_size`
            lens = torch.tensor([len(X_with_opt[batch_iter_cnt]) for batch_iter_cnt in range(len(X))]).to(device)
            max_len = max([len(X_with_opt[batch_iter_cnt]) for batch_iter_cnt in range(len(X))])
            X_with_opt = [xiter + [voc_size for _ in range(max_len - len(xiter))] for xiter in X]
            X_with_opt = torch.tensor(X_with_opt).to(device)
            
            # feed into model
            output = model(X_with_opt,lens)

            y_pd.append([item[1] for item in output.tolist()])

        # increase count
        # notice that in dataset every true answer is in the last place.
        total_cnt += batch_size
        for item in range(batch_size):
            if max([y_pd[i][item] for i in range(n_for_1)]) == y_pd[-1][item]:
                correct_cnt += 1
    print("result: correct",correct_cnt,"in",total_cnt,", rate:",correct_cnt/total_cnt)


if __name__ == "__main__":
    valid(model_path = "../ckpt/model3000", n_for_1 = 5, device = "cuda", batch_size = 50) 