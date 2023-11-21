import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from config.components import RNNConfig


class RNN(torch.nn.Module):
    """
    rnn layer
    """
    supported_types = ["RNN", "GRU", "LSTM"]

    def __init__(self, config: RNNConfig):
        super(RNN, self).__init__()
        self.rnn_type = config.rnn_type
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional

        if config.rnn_type not in self.supported_types:
            types_str = " ".join(self.supported_types)
            msg = f"Unsupported rnn init type: {config.rnn_type}."
            msg += f" Supported rnn type is: {types_str}"
            raise TypeError(msg)

        self.rnn = getattr(torch.nn, config.rnn_type)(
            config.input_size,
            config.hidden_size,
            num_layers=config.num_layers,
            bias=config.bias,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            batch_first=True
        )

    # verify for seq_lengths
    def forward(self, inputs, seq_lengths=None, init_state=None):
        if seq_lengths is not None:
            seq_lengths = seq_lengths.int()
            sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)
            sorted_inputs = inputs[indices]
            packed_inputs = pack_padded_sequence(
                sorted_inputs,
                sorted_seq_lengths.cpu(),
                batch_first=True
            )
            outputs, state = self.rnn(packed_inputs, init_state)
        else:
            outputs, state = self.rnn(inputs, init_state)

        if self.rnn_type == "LSTM":
            state = state[0]

        if self.bidirectional:  # concatenate bidirectional hidden state
            last_layers_hn = state[2 * (self.num_layers - 1):]
            last_layers_hn = torch.cat((last_layers_hn[0], last_layers_hn[1]), 1)
        else:
            last_layers_hn = state[self.num_layers - 1:]
            last_layers_hn = last_layers_hn[0]

        if seq_lengths is not None:
            # re index
            _, revert_indices = torch.sort(indices, descending=False)
            last_layers_hn = last_layers_hn[revert_indices]
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[revert_indices]
        return outputs, last_layers_hn


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim, dropout=0):
        super(AttentionLayer, self).__init__()
        self.attention_matrix = nn.Linear(input_dim, attention_dim)
        self.attention_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, inputs, seq_lens):
        u = torch.tanh(self.attention_matrix(inputs))
        attn_logits = self.attention_vector(u).squeeze(2)  # [batch_size, maxLen]

        for i, seq_len in enumerate(seq_lens):
            attn_logits[i][seq_len.item():] = -1e9
        alpha = F.softmax(attn_logits, 1).unsqueeze(1)  # [batch_size, 1, maxLen]
        context = torch.matmul(alpha, inputs).squeeze(1)  # [batch_size, emb]
        return context
    
class MRNN(torch.nn.Module):
    """
    Memory-augmented LSTM layer
    """
    supported_types = ["MRNN"]

    def __init__(self, config: RNNConfig):
        super(MRNN, self).__init__()
        self.rnn_type = config.rnn_type
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional

        if config.rnn_type not in self.supported_types:
            types_str = " ".join(self.supported_types)
            msg = f"Unsupported rnn init type: {config.rnn_type}."
            msg += f" Supported rnn type is: {types_str}"
            raise TypeError(msg)

        self.rnn = MemoryAugmentedLSTM(
            config.input_size,
            config.hidden_size,
            num_layers=config.num_layers,
            bias=config.bias,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            batch_first=True,
            forget_gate_size=config.hidden_size,
            input_gate_size=config.hidden_size,
            output_gate_size=config.hidden_size
        )

    def forward(self, inputs, seq_lengths=None, init_state=None):
        if seq_lengths is not None:
            seq_lengths = seq_lengths.int()
            sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)
            sorted_inputs = inputs[indices]
            packed_inputs = pack_padded_sequence(
                sorted_inputs,
                sorted_seq_lengths.cpu(),
                batch_first=True
            )
            outputs, state = self.rnn(packed_inputs, init_state)
        else:
            outputs, state = self.rnn(inputs, init_state)

        if self.bidirectional:
            # Concatenate bidirectional hidden state
            # state structure: (h_0, c_0) for LSTM, h_0 for others
            if self.rnn_type == "LSTM":
                # LSTM bidirectional state concatenation
                forward_state = state[0][-2:, :, :]  # (num_layers * num_directions, batch, hidden_size)
                backward_state = state[0][-2:, :, :]
                state = (torch.cat((forward_state, backward_state), dim=2),)
            else:
                # Other RNNs bidirectional state concatenation
                forward_state = state[-2:, :, :]  # (num_layers * num_directions, batch, hidden_size)
                backward_state = state[-2:, :, :]
                state = torch.cat((forward_state, backward_state), dim=2)

        return outputs, state

class MemoryAugmentedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, forget_gate_size, input_gate_size, output_gate_size, bias=True):
        super(MemoryAugmentedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_gate_size = forget_gate_size
        self.input_gate_size = input_gate_size
        self.output_gate_size = output_gate_size

        self.input_weights = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.forget_gate_memory = nn.Linear(hidden_size, forget_gate_size, bias=bias)
        self.input_gate_memory = nn.Linear(hidden_size, input_gate_size, bias=bias)
        self.output_gate_memory = nn.Linear(hidden_size, output_gate_size, bias=bias)

    def forward(self, input, hidden):
        h_prev, c_prev = hidden

        combined = self.input_weights(input) + self.hidden_weights(h_prev)

        # Split combined into gates and memory components
        input_gate, forget_gate, cell_gate, output_gate = torch.split(combined, self.hidden_size, dim=1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        cell_gate = torch.tanh(cell_gate)

        # Update the cell and hidden state
        c = (forget_gate * c_prev) + (input_gate * cell_gate)
        h = output_gate * torch.tanh(c)

        return h, c


class MemoryAugmentedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, forget_gate_size, input_gate_size, output_gate_size,
                 num_layers=1, bias=True, dropout=0, bidirectional=False, batch_first=False):
        super(MemoryAugmentedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_gate_size = forget_gate_size
        self.input_gate_size = input_gate_size
        self.output_gate_size = output_gate_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        # Create a list of LSTM cells
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = input_size if layer == 0 else hidden_size
            cell = MemoryAugmentedLSTMCell(
                input_dim, hidden_size, forget_gate_size, input_gate_size, output_gate_size, bias=bias
            )
            self.cells.append(cell)

    def forward(self, input, hidden=None):
        if self.batch_first:
            input = input.transpose(0, 1)

        if hidden is None:
            h_0 = input.new_zeros(self.num_layers * (2 if self.bidirectional else 1), input.size(1), self.hidden_size)
            hidden = (h_0,)

        h, c = hidden[0], hidden[1]

        outputs = []
        for seq in range(input.size(0)):
            x = input[seq]
            for layer in range(self.num_layers):
                cell = self.cells[layer]
                h[layer], c[layer] = cell(x, (h[layer], c[layer]))
                x = h[layer]

            outputs.append(x)

        outputs = torch.stack(outputs)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)

        return outputs, (h, c)

