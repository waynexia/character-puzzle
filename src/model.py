import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        self.attn_linear = nn.Linear(self.hidden_size,self.hidden_size).to(self.device)
        self.attn = self.attention_net
        self.out = nn.Linear(self.hidden_size,2).to(self.device)
    
    # attention net
    def attention_net(self,input,final_state,lens):
        """
        @input:         [batch size, max length, hidden size]
        @final_state:   [n_layer, batch size, hidden size]
        """

        # make mask
        maxlen = input.size(1)
        mask = torch.arange(maxlen).to(self.device)[None, :] <= lens[:, None] # <= because add answer to the begin, makes size increase

        # `fake` general method
        energy = self.attn_linear(input).to(self.device)
        hidden = final_state.transpose(0,1).to(self.device) # [batch size, n_layer, hidden size]
        attn_weights = torch.sum(hidden * energy, dim=2) # [batch size, max length]
        attn_weights[~mask] = float('-inf') # [batch size, max length]

        # softmax
        attn_weights = attn_weights.unsqueeze(1).to(self.device) # [batch size, n_layer(1), max length]
        soft_attn_weights = F.softmax(attn_weights,2).to(self.device) # [batch size, n_layer(1), max length]
        new_hidden_state = torch.bmm(soft_attn_weights,input).to(self.device) # [batch size, n_layer(1), max length]
        return new_hidden_state

    def forward(self,input,lens):
        
        # embed
        input_embedded = self.embedding(input).to(self.device)

        # pad
        input_packed = pack_padded_sequence(input_embedded, lens, batch_first=True,enforce_sorted=False).to(self.device)

        # feed into gru
        gru_output_packed, hidden = self.gru(input_packed)

        # concat gru_output with opt
        output_unpacked,_ = pad_packed_sequence(gru_output_packed, batch_first=True)

        # feed into attn
        attn_output = self.attn(output_unpacked,hidden,lens).to(self.device)

        # omit length dim
        out = self.out(attn_output).squeeze(1).to(self.device)

        # softmax
        out = F.softmax(out,dim = 1)

        return out
