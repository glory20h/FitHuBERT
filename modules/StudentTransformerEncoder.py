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
class StudentTransformerEncoderConfig(FairseqDataclass):
    
    layer_setting: TransformerSentenceEncoderLayerConfig = field(
        default=TransformerSentenceEncoderLayerConfig(),
        metadata={"help": "Default setting of TransformerSentenceEncoderLayerConfig"}
    )
    
    # layer setting after time reduction layer
    # You need to change this inside the class
    '''
    smaller_layer_setting: TransformerSentenceEncoderLayerConfig = field(
        default=TransformerSentenceEncoderLayerConfig(
            encoder_embed_dim = 384,
            encoder_ffn_embed_dim = 1536,
            encoder_attention_heads = 6 
        ),
        metadata={"help": "Time reduction layer of TransformerSentenceEncoderLayerConfig"}
    )
    '''
    encoder_layers: int = field(
        default=6,
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
    
    # Time-reduction layer
    able_tr_layer: bool = field(
        default=True,
        metadata={"help": "applying time reduction layer or not"}
    )
    
    type_of_tr_layer: str = field(
        default="fcl", # or conv1d
        metadata={"help": "type of time reduction layer"}
    )
    
    tr_conv1d_kernel_stride: str = field(
        default="(2, 2)",
        metadata={"help": "If tr is conv1d, list of kernel and stride for conv1d"}
    )
    
    tr_fcl_output_factor: int = field(
        default=2,
        metadata={"help": "Factor to reduce time length"}
    )
    
    tr_layer_floor: int = field(
        default=3,
        metadata={"help": "which floor should time reduction layer put in"}
    )

    
class StudentTransformerEncoder(nn.Module):
    def __init__(self,
                cfg: StudentTransformerEncoderConfig
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

        self.tr_fcl_output_factor = None
        if not cfg.able_tr_layer:
            tr_layer = None  
        else:
            if cfg.type_of_tr_layer == 'fcl':
                self.tr_fcl_output_factor = cfg.tr_fcl_output_factor
                # Input length will be verified first.
                tr_layer = nn.Linear(
                    self.embedding_dim * self.tr_fcl_output_factor,
                    self.embedding_dim
                )
                nn.init.xavier_uniform_(tr_layer.weight)
                
            elif cfg.type_of_tr_layer == 'conv1d':
                (kernel, stride) = eval(cfg.tr_conv1d_kernel_stride)
                tr_layer = nn.Conv1d(
                    self.embedding_dim,
                    self.embedding_dim,
                    kernel_size=kernel,
                    stride=stride
                )
            else:
                print ("Wrong type of time reduction layer.")             
        self.tr_layer = tr_layer
        
        if not cfg.able_tr_layer:
            self.layers = nn.ModuleList(
                [
                    TransformerSentenceEncoderLayer(cfg.layer_setting)
                    for _ in range(cfg.encoder_layers)
                ]
            )
        else:
            self.layers = nn.Sequential(
                nn.ModuleList(
                    [
                        TransformerSentenceEncoderLayer(cfg.layer_setting)
                        for _ in range(cfg.tr_layer_floor)
                    ],
                ),
                nn.ModuleList(
                    [
                        self.tr_layer
                    ]
                     ),
                nn.ModuleList(
                    [
                        TransformerSentenceEncoderLayer(cfg.layer_setting)
                        for _ in range(cfg.encoder_layers - cfg.tr_layer_floor)
                    ],                                
                )
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
        
        if self.tr_layer is None:
            for j, layer in enumerate(self.layers):
                dropout_probability = np.random.random()
                if not self.training or (dropout_probability > self.layerdrop):
                    x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                    if tgt_layer is not None:
                        layer_results.append((x, z))
                if j == tgt_layer:
                    r = x
                    break                
        else:
            for i, layer_block in enumerate(self.layers):
                # I write this code in this way intentionally
                # TransformerEnocder             
                if i == 0:
                    for j, layer in enumerate(layer_block):
                        dropout_probability = np.random.random()
                        if not self.training or (dropout_probability > self.layerdrop):
                            x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                            if tgt_layer is not None:
                                layer_results.append((x, z))
                        if i == tgt_layer:
                            r = x
                            break
                # Time Reduction
                elif i == 1:   
                    for j, layer in enumerate(layer_block):
                        if isinstance(layer, torch.nn.Conv1d): 
                            x = x.permute(1, 2, 0).contiguous()
                            x = layer(x)
                            x = x.permute(2, 0, 1).contiguous()
                        elif isinstance(layer, torch.nn.Linear):
                            # T x B x C
                            x = self.concat_channelwise(x)
                            x = layer(x)
                # TransformerEncoder
                elif i == 2:
                    for j, layer in enumerate(layer_block):
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
        """Upgrade a (possibly old) state dic\t for new versions of fairseq."""
        return state_dict
    
    def concat_channelwise(self, x):
        # x is shaped T x B x C
        time_length, batch, channel = x.size()
        how_many_pad = self.tr_fcl_output_factor - time_length % self.tr_fcl_output_factor 
        if how_many_pad != 0:
            zero_pad = torch.zeros([how_many_pad, batch, channel]).cuda()
            x = torch.cat([x, zero_pad], dim = 0)
        time_length += how_many_pad

        result = torch.tensor([]).cuda()
        for i in range (time_length // self.tr_fcl_output_factor):
            j = 0
            tensor_to_concat = torch.tensor([]).cuda()
            while (j < self.tr_fcl_output_factor):
                # B x (C * factor)
                tensor_to_concat = torch.cat((tensor_to_concat,
                                              x[self.tr_fcl_output_factor * i + j, :, :]), dim = 1)
                j += 1
            tensor_to_concat = tensor_to_concat.unsqueeze(0)
            result = torch.cat([result, tensor_to_concat], dim = 0)

        return result         