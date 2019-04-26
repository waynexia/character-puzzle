import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
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
    def __init__(self, batch_size, voc_size, hidden_size, device,n_layers=1, dropout=0):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.device = device
        self.batch_size = batch_size
        self.embedding = nn.Embedding(voc_size+1, hidden_size).to(self.device)
        self.n_layers = n_layers
        self.dropout = dropout
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,batch_first=True,dropout=(0 if n_layers == 1 else dropout)).to(self.device)
        self.attn = Attn(method = "general", hidden_size = hidden_size).to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(self.hidden_size,2).to(self.device)

    def forward(self,input,opt,lens):
        
        input_embedded = self.embedding(input).to(self.device)
        opt_embedded = self.embedding(opt).to(self.device)
        #opt_embedded = opt_embedded.unsqueeze(0)

        # add batch dim
        #input_embedded = input_embedded.unsqueeze(1)
        opt_embedded = opt_embedded.unsqueeze(1).to(self.device)

        #print("input before packed:",input_embedded.shape)
        input_packed = pack_padded_sequence(input_embedded, lens, batch_first=True,enforce_sorted=False).to(self.device)
        #print("input after packed :",input_packed.shape)

        
        gru_output_packed, hidden = self.gru(input_packed)
        #print("gru output :" ,gru_output_packed.shape)

        # concat gru_output with opt
        output_unpacked,_ = pad_packed_sequence(gru_output_packed, batch_first=True)
        #print("output after unpacked :",output_unpacked.shape)

        #print("things to concat: ",output_unpacked.shape," another : ",opt_embedded.shape)
        concat_output = torch.cat((opt_embedded, output_unpacked),dim = 1).to(self.device)
        #print("concated output :",concat_output.shape)

        # process each item in batch independently
        _attn_output = list()
        for i in range(self.batch_size):
            #print("lens: ",lens)
            #print("a b: ",concat_output[i][:lens[i]].unsqueeze(0).shape,hidden.shape)
            attn_weight = self.attn(concat_output[i][:lens[i]].unsqueeze(0),hidden.transpose(0,1)[i].unsqueeze(0)).to(self.device)
            #print("attn weight: ", attn_weight.shape)
            attn_weight = F.softmax(attn_weight,2).to(self.device)

            #print("two things to bmm: ",attn_weight.transpose(0,2).shape,concat_output[i][:lens[i]].unsqueeze(0).shape)
            context = attn_weight.transpose(0,2).bmm(concat_output[i][:lens[i]].unsqueeze(0)).to(self.device)
            #print("context: ",context.shape)
            #print(context)
            
            _attn_output.append(context)

        attn_output = torch.cat(_attn_output).to(self.device)
        #print("each item concat: ",attn_output.shape)
        
        sigmoid_output = self.sigmoid(attn_output).to(self.device)
        #print("sigmoid output: ",sigmoid_output.shape)

        out = self.out(sigmoid_output).squeeze(1).to(self.device)
        #print("linear output: ",out.shape)

        #out = F.softmax(out)
        #exit()

        return out




