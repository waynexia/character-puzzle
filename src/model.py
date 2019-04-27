import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch
import torch.nn.functional as F
import numpy as np
import sys

"""
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

# batch attn
# from https://github.com/kevinlu1211/pytorch-batch-luong-attention/blob/master/models/luong_attention_batch/luong_attention_batch.py
class BatchAttn(nn.Module):
    """
    Note here that we are only implementing the 'general' method as denoted in the paper
    """

    def __init__(self, hidden_size, use_cuda):
        super().__init__()
        self.hidden_size = hidden_size
        self.general_weights = torch.autograd.Variable(torch.randn(hidden_size, hidden_size))
        self.use_cuda = use_cuda
        if use_cuda:
            self.general_weights = self.general_weights.cuda()

    def forward(self,
                encoder_outputs,
                encoder_outputs_length,
                decoder_outputs,
                decoder_outputs_length):
        """
        :param encoder_outputs: max_encoder_length, batch_size, hidden_size
        :param encoder_outputs_length: batch_size
        :param decoder_outputs: max_decoder_length, batch_size, hidden_size
        :param decoder_outputs_length: batch_size
        :return: attention_aware_output
        """

        # (batch_size, max_decoder_length, hidden_size)
        decoder_outputs = torch.transpose(decoder_outputs, 0, 1)

        # (batch_size, hidden_size, max_encoder_length)
        #encoder_outputs = encoder_outputs.permute(1, 2, 0)

        # (batch_size, max_encoder_length, max_decoder_length
        print(decoder_outputs.shape,self.general_weights.shape,encoder_outputs.shape)
        score = torch.bmm(decoder_outputs @ self.general_weights, encoder_outputs)

        (attention_mask,
         max_enc_outputs_length,
         max_dec_outputs_length) = self.attention_sequence_mask(encoder_outputs_length, decoder_outputs_length)
        masked_score = score + attention_mask
        weights_flat = F.softmax(masked_score.view(-1, max_enc_outputs_length))
        weights = weights_flat.view(-1, max_dec_outputs_length, max_enc_outputs_length)

        return weights

    def sequence_mask(self,sequence_length, max_len=None, use_cuda=False):
        if isinstance(sequence_length, np.ndarray):
            sequence_length = torch.autograd.Variable(torch.from_numpy(sequence_length))
        elif isinstance(sequence_length, list):
            sequence_length = torch.autograd.Variable(torch.from_numpy(np.array(sequence_length)))

        if max_len is None:
            max_len = sequence_length.data.max()

        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = torch.autograd.Variable(seq_range_expand)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1)
                            .expand_as(seq_range_expand))
        mask = seq_range_expand < seq_length_expand
        if use_cuda:
            mask = mask.cuda()
        return mask

    def attention_sequence_mask(self, encoder_outputs_length, decoder_outputs_length):
        batch_size = len(encoder_outputs_length)
        max_encoder_outputs_length = max(encoder_outputs_length)
        max_decoder_outputs_length = max(decoder_outputs_length)

        encoder_sequence_mask = self.sequence_mask(encoder_outputs_length, use_cuda=self.use_cuda)
        encoder_sequence_mask_expand = (encoder_sequence_mask
                                        .unsqueeze(1)
                                        .expand(batch_size,
                                                max_decoder_outputs_length,
                                                max_encoder_outputs_length))

        decoder_sequence_mask = self.sequence_mask(decoder_outputs_length, use_cuda=self.use_cuda)
        decoder_sequence_mask_expand = (decoder_sequence_mask
                                        .unsqueeze(2)
                                        .expand(batch_size,
                                                max_decoder_outputs_length,
                                                max_encoder_outputs_length))
        attention_mask = (encoder_sequence_mask_expand *
                          decoder_sequence_mask_expand).float()
        attention_mask = (attention_mask - 1) * sys.maxsize
        return (attention_mask,
                max_encoder_outputs_length,
                max_decoder_outputs_length)
"""

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
        #self.attn = Attn(method = "general", hidden_size = hidden_size).to(self.device) # raw attn
        #self.attn = BatchAttn(hidden_size = self.hidden_size,use_cuda = True) # batch attn (not work)
        self.attn_linear = nn.Linear(self.hidden_size,self.hidden_size).to(self.device)
        self.attn = self.attention_net # dont know work or not
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(self.hidden_size,2).to(self.device)
    
    # attention net
    def attention_net(self,input,final_state,lens):
        """
        @input:         [batch size, max length, hidden size]
        @final_state:   [n_layer, batch size, hidden size]
        """

        # make mask
        maxlen = input.size(1)
        mask = torch.arange(maxlen).to(self.device)[None, :] < lens[:, None]

        #print(input.shape,final_state.shape)
        energy = self.attn_linear(input)
        hidden = final_state.transpose(0,1).cuda() # [batch size, n_layer, hidden size]
        attn_weights = torch.sum(hidden * energy, dim=2) # [batch size, max length]
        attn_weights[~mask] = float('-inf') # [batch size, max length]
        #print(attn_weights.shape)
        attn_weights = attn_weights.unsqueeze(1).to(self.device) # [batch size, n_layer(1), max length]
        #print(attn_weights.shape)
        #print(attn_weights)
        soft_attn_weights = F.softmax(attn_weights,2).cuda() # [batch size, n_layer(1), max length]
        #print(soft_attn_weights)
        #print(soft_attn_weights)
        new_hidden_state = torch.bmm(soft_attn_weights,input).cuda() # [batch size, n_layer(1), max length]
        return new_hidden_state

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
        # [batch, max_length, hidden_size]
        concat_output = torch.cat((opt_embedded, output_unpacked),dim = 1).to(self.device)
        #print("concated output :",concat_output.shape)

        # process each item in batch independently
        """_attn_output = list()
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
            
            _attn_output.append(context)"""
        
        #attn_weight = self.attn(concat_output,hidden)#
        #attn_output = torch.bmm(attn_weight,concat_output.transpose(0,1))#
        attn_output = self.attn(concat_output,hidden,lens)

        """attn_output = torch.cat(_attn_output).to(self.device)"""
        #print("each item concat: ",attn_output.shape)
        
        sigmoid_output = self.sigmoid(attn_output).to(self.device)
        #print("sigmoid output: ",sigmoid_output.shape)

        out = self.out(sigmoid_output).squeeze(1).to(self.device)
        #print("linear output: ",out.shape)

        #out = F.softmax(out)

        return out




