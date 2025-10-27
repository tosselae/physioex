import torch
import torch.nn as nn

from physioex.train.networks.prototype import ProtoSleepModule, ProtoSleepNet
from physioex.train.networks.seqsleepnet import LearnableFilterbank


class ProtoSeqSleepNet(ProtoSleepModule):
    def __init__(self, module_config: dict):
        super(ProtoSeqSleepNet, self).__init__(NN(module_config), module_config)


class NN(ProtoSleepNet):
    def __init__(self, module_config: dict):
        super(NN, self).__init__(module_config)

        self.e_encoder = EpochEncoder()
        self.s_encoder = SequenceEncoder()

    def epoch_encoder(self, x):
        return self.e_encoder(x)

    def sequence_encoder(self, x):
        return self.s_encoder(x)

    def eval_on_night(self, inputs, L=21):
        batch_size, night_length, n_channels, T, F = inputs.size()

        x, mcy = self.f_ExE(inputs, return_mcy=True)

        # prototyping
        p, commit_loss = self.f_P(x, return_commit_loss=True)
        proto_y = self.clf(p.reshape(batch_size * night_length, -1)).reshape(
            batch_size, night_length, -1
        )

        x = self.f_ExS(p)
        y = self.clf(x.reshape(batch_size * night_length, -1)).reshape(
            batch_size, night_length, -1
        )  # (batch_size, night_length, 5)

        self.mcy = mcy
        self.commit_loss = commit_loss
        self.proto_y = proto_y

        return p, y


class ChannelEncoder(nn.Module):
    def __init__(self):
        super(ChannelEncoder, self).__init__()
        self.filtbank = LearnableFilterbank(in_chan=1, nfilt=32)

        self.birnn = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
        )

        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        batch, T, F = x.size()

        x = self.filtbank(x)

        # lstm are not implemented for bfloat16
        # convert to float32 for processing
        dtype = x.dtype
        x, _ = self.birnn(x.to(torch.float32))
        x = x.to(dtype)

        x = self.norm(x)

        return x


class EpochEncoder(nn.Module):
    def __init__(self):
        super(EpochEncoder, self).__init__()

        self.eeg_encoder = ChannelEncoder()
        self.eog_encoder = ChannelEncoder()
        self.emg_encoder = ChannelEncoder()

    def forward(self, x):
        batch, nchan, T, F = x.size()

        eeg = self.eeg_encoder(x[:, 0])
        eog = self.eog_encoder(x[:, 1])
        emg = self.emg_encoder(x[:, 2])

        eeg = eeg.reshape(batch, 1, T, -1)
        eog = eog.reshape(batch, 1, T, -1)
        emg = emg.reshape(batch, 1, T, -1)

        x = torch.cat((eeg, eog, emg), dim=1)

        return x


class SequenceEncoder(nn.Module):
    def __init__(self):
        super(SequenceEncoder, self).__init__()

        self.encoder = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
        )

        # Initialize GRU as a residual layer to be identity at training start
        self.encoder = _initialize_residual_gru(self.encoder)

    def _initialize_residual_gru(self):
        """
        Initialize GRU to act as identity at training start for residual connection.
        For residual: output = input + gru(input)
        We want gru(input) ≈ 0 initially, so output ≈ input

        Note: This method is deprecated. Use the module-level _initialize_residual_gru() instead.
        """
        return _initialize_residual_gru(self.encoder)

    def forward(self, x):
        batch, L, _ = x.size()

        dtype = x.dtype
        x, _ = self.encoder(x.to(torch.float32))
        x = x.to(dtype)
        return x


def _initialize_residual_gru(gru: nn.GRU):
    """
    Utility function to initialize any GRU for residual connections.

    Args:
        gru: nn.GRU module to initialize

    Returns:
        gru: The initialized GRU module
    """
    for name, param in gru.named_parameters():
        if "weight_hh" in name:
            # Hidden-to-hidden weights: orthogonal with small gain
            nn.init.orthogonal_(param.data, gain=0.01)
        elif "weight_ih" in name:
            # Input-to-hidden weights: Xavier uniform with small gain
            nn.init.xavier_uniform_(param.data, gain=0.01)
        elif "bias" in name:
            # Zero initialization for biases
            nn.init.constant_(param.data, 0.0)

            # GRU-specific bias initialization for residual behavior
            if param.data.numel() >= param.data.size(0) * 2:  # has gates
                hidden_size = param.data.size(0) // 3  # GRU has 3 gates per direction

                # Reset gate bias: small positive (encourage forgetting initially)
                param.data[hidden_size : 2 * hidden_size].fill_(0.1)

                # Update gate bias: small negative (encourage ignoring new input)
                param.data[0:hidden_size].fill_(-0.1)

                # New gate bias: keep at zero (neutral)
                param.data[2 * hidden_size : 3 * hidden_size].fill_(0.0)

    return gru
