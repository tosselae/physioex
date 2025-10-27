import torch
import torch.nn as nn

from physioex.train.networks.prototype import ProtoSleepModule, ProtoSleepNet
from physioex.train.networks.sleeptransformer import PositionalEncoding

module_config = dict()


class ProtoSleepTransformerNet(ProtoSleepModule):
    def __init__(self, module_config: dict = module_config):
        super(ProtoSleepTransformerNet, self).__init__(NN(module_config), module_config)


class NN(ProtoSleepNet):
    def __init__(self, module_config=module_config):
        super(NN, self).__init__(module_config)

        self.e_encoder = EpochEncoder()
        self.s_encoder = SequenceEncoder()

    def epoch_encoder(self, x):
        return self.e_encoder(x)

    def sequence_encoder(self, x):
        return self.s_encoder(x)


class EpochEncoder(nn.Module):
    def __init__(self):
        super(EpochEncoder, self).__init__()

        self.pe = PositionalEncoding(128)

        t_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            t_layer,
            num_layers=4,
        )

    def forward(self, x):
        batch, nchan, T, F = x.size()
        x = x.reshape(batch * nchan, T, F)[..., :128]

        x = self.pe(x)
        x = self.encoder(x)

        return x.reshape(batch, nchan, T, -1)  # (batch, nchan, 128)


class SequenceEncoder(nn.Module):
    def __init__(self):
        super(SequenceEncoder, self).__init__()

        self.pe = PositionalEncoding(128)
        t_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(t_layer, num_layers=4)

        # Initialize sequence encoder as a residual layer to be identity at training start
        # This prevents the encoder from affecting the input initially, allowing gradual learning
        self._initialize_residual_transformer()

    def _initialize_residual_transformer(self):
        """
        Initialize the TransformerEncoder to act as identity at training start.
        For residual connection: output = input + encoder(input)
        We want encoder(input) ≈ 0 initially, so output ≈ input
        """
        for layer in self.encoder.layers:
            # Initialize feedforward layers to zero
            torch.nn.init.constant_(layer.linear1.weight, 0.0)
            torch.nn.init.constant_(layer.linear1.bias, 0.0)
            torch.nn.init.constant_(layer.linear2.weight, 0.0)
            torch.nn.init.constant_(layer.linear2.bias, 0.0)

            # Initialize attention output projection to zero
            torch.nn.init.constant_(layer.self_attn.out_proj.weight, 0.0)
            torch.nn.init.constant_(layer.self_attn.out_proj.bias, 0.0)

            # Initialize Q, K, V projections with small values for stability
            torch.nn.init.normal_(layer.self_attn.in_proj_weight, mean=0.0, std=0.01)
            if layer.self_attn.in_proj_bias is not None:
                torch.nn.init.constant_(layer.self_attn.in_proj_bias, 0.0)

            # Layer norms keep default initialization (weight=1, bias=0)

    def _initialize(self):
        return self._initialize_residual_transformer()

    def forward(self, x):
        batch, L, _ = x.size()
        x = self.pe(x)
        x = self.encoder(x)
        return x

        # EpochEncoder does not need residual initialization as it's used directly
        # without residual connections in the f_ExE method
