import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Tuple

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
from fairseq import utils

@dataclass
class TransformerSentenceEncoderLayerConfig(FairseqDataclass):
    encoder_embed_dim: int = field(
        default=768,
        metadata={"help": "encoder embedding dimension"}
    )
    
    encoder_ffn_embed_dim: int = field(
        default=3072,
        metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, # 8 for model init code
        metadata={"help": "num encoder attention heads"}
    )
    
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"}
    )
    
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"}
    )
    
    activation_dropout: float = field(
        default=0.0, # 0.1 for model init code
        metadata={"help": "dropout probability after activation in FFN"}
    )
    
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", # relu for model init code
        metadata={"help": "activation function to use"}
        # relu, gelu, gelu_fast, gelu_accurate, tanh, linear
    )
    
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"}
    )
    
    
class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        cfg: TransformerSentenceEncoderLayerConfig
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = cfg.encoder_embed_dim
        dropout = cfg.dropout
        self.activation_dropout = cfg.activation_dropout
        self.layer_norm_first = cfg.layer_norm_first
        num_attention_heads = cfg.encoder_attention_heads
        attention_dropout = cfg.attention_dropout
        activation_fn = cfg.activation_fn
        ffn_embedding_dim = cfg.encoder_ffn_embed_dim
        
        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn