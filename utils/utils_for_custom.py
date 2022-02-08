from argparse import Namespace
import contextlib
import yaml
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from omegaconf import MISSING, II, open_dict
from typing import Any, Optional
from datetime import datetime
from pytz import timezone

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.tasks import FairseqTask
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer,
)

from modules.Wav2Vec2Model import Wav2Vec2Config
from modules.CustomWav2Vec2Model import CustomWav2Vec2Config


def convert_dict_to_config(cfg):
    # Input : cfg; Config for wav2vec2 model
    config = Wav2Vec2Config()
    
    conv_layer_config = config.conv_layer_setting
    encoder_config = config.encoder_setting
    encoder_layer_config = encoder_config.layer_setting
    
    # Feature Extractor Config
    conv_layer_config.extractor_mode = cfg.extractor_mode
    conv_layer_config.conv_feature_layers = cfg.conv_feature_layers
    conv_layer_config.conv_bias = cfg.conv_bias
    conv_layer_config.conv_dropout = 0.0 # by default
    
    # Encoder Layer each Config
    encoder_layer_config.encoder_embed_dim = cfg.encoder_embed_dim
    encoder_layer_config.encoder_ffn_embed_dim = cfg.encoder_ffn_embed_dim
    encoder_layer_config.encoder_attention_heads = cfg.encoder_attention_heads
    encoder_layer_config.dropout = cfg.dropout
    encoder_layer_config.attention_dropout = cfg.attention_dropout
    encoder_layer_config.activation_dropout = cfg.activation_dropout
    encoder_layer_config.activation_fn = cfg.activation_fn
    encoder_layer_config.layer_norm_first = cfg.layer_norm_first
    
    # Encoder Config
    encoder_config.layer_setting = encoder_layer_config
    encoder_config.encoder_layers = cfg.encoder_layers
    encoder_config.conv_pos = cfg.conv_pos
    encoder_config.conv_pos_groups = cfg.conv_pos_groups
    encoder_config.encoder_layerdrop = cfg.encoder_layerdrop
    
    # Wav2vec2 Model Config
    config.conv_layer_setting = conv_layer_config
    config.encoder_setting = encoder_config
    config.dropout_input = cfg.dropout_input
    config.dropout_features = cfg.dropout_features
    config.final_dim = cfg.final_dim
    config.logit_temp = cfg.logit_temp
    config.quantize_targets = cfg.quantize_targets
    config.quantize_input = cfg.quantize_input
    config.same_quantizer = cfg.same_quantizer
    config.target_glu = cfg.target_glu
    config.feature_grad_mult = cfg.feature_grad_mult
    config.quantizer_depth = cfg.quantizer_depth
    config.quantizer_factor = cfg.quantizer_factor
    config.latent_vars = cfg.latent_vars
    config.latent_groups = cfg.latent_groups
    config.latent_dim = cfg.latent_dim
    config.mask_length = cfg.mask_length
    config.mask_prob = cfg.mask_prob
    config.mask_selection = cfg.mask_selection
    config.mask_other = cfg.mask_other
    config.no_mask_overlap = cfg.no_mask_overlap
    config.mask_channel_length = cfg.mask_channel_length
    config.mask_min_space = cfg.mask_min_space
    config.mask_channel_prob = cfg.mask_channel_prob
    config.mask_channel_before = cfg.mask_channel_before
    config.mask_channel_selection = cfg.mask_channel_selection
    config.mask_channel_other = cfg.mask_channel_other
    config.no_mask_channel_overlap = cfg.no_mask_channel_overlap
    config.mask_channel_min_space = cfg.mask_channel_min_space
    config.num_negatives = cfg.num_negatives
    config.negatives_from_everywhere = cfg.negatives_from_everywhere
    config.cross_sample_negatives = cfg.cross_sample_negatives
    config.codebook_negatives = cfg.codebook_negatives
    config.latent_temp = cfg.latent_temp
    
    return config


def convert_dict_to_custom_config(cfg):
    # Input : cfg; Config for CustomWav2Vec2 model
    config = CustomWav2Vec2Config()
    
    conv_layer_config = config.conv_layer_setting
    encoder_config = config.encoder_setting
    encoder_layer_config = encoder_config.layer_setting
    
    # Feature Extractor Config
    conv_layer_config.extractor_mode = cfg.extractor_mode
    conv_layer_config.conv_feature_layers = cfg.conv_feature_layers
    conv_layer_config.conv_bias = cfg.conv_bias
    conv_layer_config.conv_dropout = 0.0 # by default
    
    # Encoder Layer each Config
    encoder_layer_config.encoder_embed_dim = cfg.encoder_embed_dim
    encoder_layer_config.encoder_ffn_embed_dim = cfg.encoder_ffn_embed_dim
    encoder_layer_config.encoder_attention_heads = cfg.encoder_attention_heads
    encoder_layer_config.dropout = cfg.dropout
    encoder_layer_config.attention_dropout = cfg.attention_dropout
    encoder_layer_config.activation_dropout = cfg.activation_dropout
    encoder_layer_config.activation_fn = cfg.activation_fn
    encoder_layer_config.layer_norm_first = cfg.layer_norm_first
    
    # Encoder Config
    encoder_config.layer_setting = encoder_layer_config
    encoder_config.encoder_layers = cfg.encoder_layers
    encoder_config.conv_pos = cfg.conv_pos
    encoder_config.conv_pos_groups = cfg.conv_pos_groups
    encoder_config.encoder_layerdrop = cfg.encoder_layerdrop
    # Please define time-reduction layer explicitly in train.py
    
    # Wav2vec2 Model Config
    config.conv_layer_setting = conv_layer_config
    config.encoder_setting = encoder_config
    config.dropout_input = cfg.dropout_input
    config.dropout_features = cfg.dropout_features
    config.final_dim = cfg.final_dim
    config.logit_temp = cfg.logit_temp
    # config.quantize_targets = cfg.quantize_targets
    # config.quantize_input = cfg.quantize_input
    # config.same_quantizer = cfg.same_quantizer
    config.target_glu = cfg.target_glu
    config.feature_grad_mult = cfg.feature_grad_mult
    # config.quantizer_depth = cfg.quantizer_depth
    # config.quantizer_factor = cfg.quantizer_factor
    # config.latent_vars = cfg.latent_vars
    # config.latent_groups = cfg.latent_groups
    # config.latent_dim = cfg.latent_dim
    config.mask_length = cfg.mask_length
    config.mask_prob = cfg.mask_prob
    config.mask_selection = cfg.mask_selection
    config.mask_other = cfg.mask_other
    config.no_mask_overlap = cfg.no_mask_overlap
    config.mask_channel_length = cfg.mask_channel_length
    config.mask_min_space = cfg.mask_min_space
    config.mask_channel_prob = cfg.mask_channel_prob
    # config.mask_channel_before = cfg.mask_channel_before
    config.mask_channel_selection = cfg.mask_channel_selection
    config.mask_channel_other = cfg.mask_channel_other
    config.no_mask_channel_overlap = cfg.no_mask_channel_overlap
    config.mask_channel_min_space = cfg.mask_channel_min_space
    # config.num_negatives = cfg.num_negatives
    # config.negatives_from_everywhere = cfg.negatives_from_everywhere
    # config.cross_sample_negatives = cfg.cross_sample_negatives
    # config.codebook_negatives = cfg.codebook_negatives
    config.latent_temp = cfg.latent_temp
    
    return config

# Make yaml file with given name and config dataclass
def dump_yaml(cfg, yaml_dict):
    
    # cfg: updated distiller config dataclass (= student_config)
    # yaml_file: dumping yaml file (= YAML_CFG)
    distiller = dict()
    
    for attr in dir(cfg):
        if not callable(getattr(cfg, attr)) and not attr.startswith("_"):
            distiller[attr] = getattr(cfg, attr)

    dump_dict = yaml_dict

    for key in distiller:
        if key in ['activation_fn', 'extractor_mode', 'layer_type']:
            dump_dict['distiller'][key] = str(distiller[key])
        else:
            dump_dict['distiller'][key] = distiller[key]

    # name as current time
    name = datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d_%Hh%Mm%Ss')

    with open('./data/log/' + name + '.yaml', 'w') as f:
        yaml.dump(dump_dict, f, sort_keys = False)
    
    return dump_dict