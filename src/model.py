import torch.nn as nn

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
    def __init__(self, hidden_size, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.embedding(hidden_size)
        self.n_layers = n_layers
        self.dropout = dropout
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,dropout=(0 if n_layers == 1 else dropout))
        self.attn = Attn(method = "general", hidden_size = hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self,input,input_len,gd,options):
        input_embedded = self.embedding(input)
        gd_embedded = self.embedding(gd)

        gru_output, hidden = self.gru(input_embedded)

        # concat gru_output with gd
        concat_output = torch.cat((gru_output, gd_embedded), 1)

        attn_output = self.attn(concat_output,hidden)

        sigmoid_output = self.sigmoid(attn_output)

        return sigmoid_output




