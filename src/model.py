import torch.nn as nn
import torch
import torch.nn.functional as F

# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class Encoder(nn.Module):
    def __init__(self, voc_size, hidden_size, n_layers=1, dropout=0):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(voc_size, hidden_size)
        self.n_layers = n_layers
        self.dropout = dropout
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,dropout=(0 if n_layers == 1 else dropout))
        self.attn = Attn(method = "general", hidden_size = hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(self.hidden_size,2)

    def temp_attn(self,input,hidden):
        #hidden = hidden.squeeze(0)
        #attn_weight = torch.bmm(input.unsqueeze(1),hidden.unsqueeze(2))
        print(input.shape)
        print(hidden.shape)
        attn_weight = torch.bmm(input.squeeze(1),hidden.squeeze(1))
        soft_attn_weight = F.softmax(attn_weight,1)
        new_hidden_stat = torch.bmm(input.unsqueeze(1).transpose(1,2),soft_attn_weight.unsqueeze(2))
        return new_hidden_stat

    def forward(self,input,opt):
        input_embedded = self.embedding(input)
        opt_embedded = self.embedding(opt)
        opt_embedded = opt_embedded.unsqueeze(0)

        # add batch dim
        input_embedded = input_embedded.unsqueeze(1)
        opt_embedded = opt_embedded.unsqueeze(0)

        gru_output, hidden = self.gru(input_embedded)

        # concat gru_output with opt
        concat_output = torch.cat((gru_output, opt_embedded))

        #print(concat_output.shape)
        #print(hidden.shape)
        attn_weight = self.attn(concat_output,hidden)
        context = attn_weight.bmm(hidden.transpose(0,1))

        concat_output = concat_output.squeeze(0)
        context = context.squeeze(1)
        attn_output = torch.cat((concat_output,context),1)


        #attn_output = attn_output.squeeze()
        #attn_output = self.temp_attn(concat_output,hidden)
        #print(attn_output.shape)

        sigmoid_output = self.sigmoid(attn_output)

        out = self.out(sigmoid_output).unsqueeze(0)
        #print(out.shape)

        return out




