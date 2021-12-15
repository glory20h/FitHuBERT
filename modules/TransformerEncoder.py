import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Tuple

import math
import torch.nn.functional as F
import numpy as np

from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
)

from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .TransformerSentenceEncoderLayer import TransformerSentenceEncoderLayer, TransformerSentenceEncoderLayerConfig

@dataclass
class TransformerEncoderConfig(FairseqDataclass):
    
    layer_setting: TransformerSentenceEncoderLayerConfig = field(
        default=TransformerSentenceEncoderLayerConfig(),
        metadata={"help": "Default setting of TransformerSentenceEncoderLayerConfig"}
    )
    
    encoder_layers: int = field(
        default=12,
        metadata={"help": "num encoder layers in the transformer"}
    )
    
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a transformer layer"}
    )
    
class TransformerEncoder(nn.Module):
    def __init__(self,
                cfg: TransformerEncoderConfig
                ):
        
        super().__init__()
        
        args = cfg.layer_setting
        
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=cfg.conv_pos,
            padding=cfg.conv_pos // 2,
            groups=cfg.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (cfg.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(cfg.conv_pos), nn.GELU())

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    cfg.layer_setting
                )
                for _ in range(cfg.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = cfg.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, padding_mask=None, tgt_layer=None):

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                if tgt_layer is not None:
                    layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict