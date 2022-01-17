import numpy as np
import torch
import torch.nn as nn
import math

from itertools import groupby
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from omegaconf import MISSING, II, open_dict

from argparse import Namespace
import contextlib
import copy

from torch.nn.utils.rnn import pad_sequence
from fairseq import quantization_utils
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.tasks.audio_finetuning import AudioFinetuningTask
from fairseq.dataclass.utils import convert_namespace_to_omegaconf, merge_with_parent
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model, Wav2Vec2Config
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc, Wav2Vec2CtcConfig

from modules.CustomWav2Vec2Model import CustomWav2Vec2Model

# Alternative for Wav2VecEncoder
class CustomStudentModel(nn.Module):
    def __init__(self, config, task_agnostic=True, pred_layer_id=[-1]):
        super(CustomStudentModel, self).__init__()
        self.config = config
        self.student_model = CustomWav2Vec2Model(config)
        self.task_agnostic = task_agnostic

        # TODO: Use "proper" projection head -> take a look at s3prl
        # encoder_embed_dim = config.encoder_setting.layer_setting.encoder_embed_dim
        # n_tasks = len(pred_layer_id)

        # inter_dim = config.proj_head_inter_dim
        # inter_dim = inter_dim if inter_dim > 0 else final_emb_size
        
        # self.proj_head = nn.Sequential(
        #     nn.Linear(encoder_embed_dim, inter_dim * n_tasks),
        #     nn.GELU(),
        #     SplitLinear(inter_dim, n_tasks, config.proj_head_final_dim),
        # )

        self.proj_heads = [ProjectionHead(config.encoder_setting.layer_setting.encoder_embed_dim) for _ in pred_layer_id]

        if not self.task_agnostic:
            self.final_dropout = nn.Dropout(config.final_dropout)
            self.final_proj = Linear(config.encoder_setting.layer_setting.encoder_embed_dim, config.targ_d)
        
    def forward(self, src, padding_mask=None, layer=100):
        result = self.student_model.extract_features(source=src, padding_mask=padding_mask, layer=layer)

        if not self.task_agnostic:
            x = result['x'].transpose(0, 1)
            x = self.final_dropout(x)
            x = self.final_proj(x)
        else:
            x = result['x']

        # Get output from projection heads
        projections = [proj_head(result['layer_results'][-1][0]) for proj_head in self.proj_heads]

        # TODO: get output from 'proper' projection head used in distilhubert
        
        return {
            "encoder_out": x,  # T x B x C
            "padding_mask": result["padding_mask"],  # B x T,
            "layer_results": result["layer_results"],
            "tr_layer_results": result["tr_layer_results"],
            "projections": projections
        }

    def init_teacher_conv_layers(self, teacher_model):
        # TODO : Adapt code for cases where it's not task-agnostic
        self.student_model.feature_extractor.load_state_dict(teacher_model.w2v_encoder.w2v_model.feature_extractor.state_dict())
        self.student_model.post_extract_proj.load_state_dict(teacher_model.w2v_encoder.w2v_model.post_extract_proj.state_dict())

    def init_teacher_encoder_layers(self, teacher_model, n_layers):
        # TODO : Adapt code for cases where it's not task-agnostic
        self.student_model.encoder.pos_conv.load_state_dict(teacher_model.w2v_encoder.w2v_model.encoder.pos_conv.state_dict())

        assert n_layers <= self.config.encoder_setting.encoder_layers

        for i in range(n_layers):
            self.student_model.encoder.layers[i].load_state_dict(teacher_model.w2v_encoder.w2v_model.encoder.layers[i].state_dict())


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim):
        super(ProjectionHead, self).__init__()
        self.layer = nn.Sequential(
                        nn.Linear(embedding_dim, embedding_dim),
                        nn.GELU(),
                        nn.Linear(embedding_dim, embedding_dim),
                    )
        
    def forward(self, x):
        return self.layer(x)


class SplitLinear(nn.Module):
    """Split Linear Layer"""

    def __init__(self, in_dim, in_split, out_dim):
        super().__init__()

        self.in_dim = in_dim  # Din
        self.in_split = in_split  # N
        self.out_dim = out_dim  # Dout

        if in_split > 1:
            weight = torch.zeros((self.in_split, self.in_dim, self.out_dim))
            self.weight = nn.Parameter(weight, requires_grad=True)
            nn.init.uniform_(self.weight, -(self.in_dim ** -0.5), self.in_dim ** -0.5)

            bias = torch.zeros((1, 1, self.in_split, self.out_dim))
            self.bias = nn.Parameter(bias, requires_grad=True)
            nn.init.uniform_(self.bias, -(self.in_dim ** -0.5), self.in_dim ** -0.5)
        else:
            self.layer = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x:torch.Tensor):
        # x: shape = B x T x NDin

        if self.in_split == 1:
            return self.layer(x)
        else:
            x = x.reshape(x.shape[0], x.shape[1], self.in_split, 1, self.in_dim)
            # x: B x T x N x 1 x Din

            out = torch.einsum("...klm,kmn->...kln", x, self.weight).squeeze(3)
            # out: B x T x N x Dout
            out = out + self.bias

            return out.reshape(x.shape[0], x.shape[1], -1) # -> B x T x NDout ?


class DataCollatorWithPadding:
    def __call__(self, features: List[Dict[str, Any]]):
        input_features = [feature[0][0] for feature in features]
        input_features_lens = [len(x) for x in input_features]
        input_features_lens = torch.LongTensor(input_features_lens)
        src = pad_sequence(input_features, batch_first=True, padding_value=0)

        mask = torch.zeros(src.shape)
        # replace one for padding
        for idx in range(src.shape[0]):
            mask[idx, input_features_lens[idx]:] = 1

        labels = [feature[2] for feature in features]
        
        return {'src': src, 'mask': mask, 'labels': labels}


class Decoder:
    def __init__(self):
        self.dict = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, 
            "T": 6, "A": 7, "O": 8, "N": 9, "I": 10, "H": 11, "S": 12, 
            "R": 13, "D": 14, "L": 15, "U": 16, "M": 17, "W": 18, "C": 19, 
            "F": 20, "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25, "K": 26, 
            "'": 27, "X": 28, "J": 29, "Q": 30, "Z": 31}
            
        self.look_up = np.asarray(list(self.dict.keys()))

    def decode(self, ids):
        converted_tokens = self.look_up[ids]
        fused_tokens = [tok[0] for tok in groupby(converted_tokens)]
        output = ' '.join(''.join(''.join(fused_tokens).split("<s>")).split("|")).rstrip()
        return output
 

class CTCSequenceConverter:
    def __init__(self, return_type="pt"):
        self.return_type = return_type
        
    def __call__(self, ids):
        if self.return_type == "pt":
            return torch.tensor([tok[0] for tok in groupby(ids) if tok[0] != 0])
        
        return [tok[0] for tok in groupby(ids) if tok[0] != 0]


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def load_model(filename, arg_overrides: Optional[Dict[str, Any]] = None):

    state = load_checkpoint_to_cpu(filename, arg_overrides)

    if "args" in state and state["args"] is not None:
        cfg = convert_namespace_to_omegaconf(state["args"])
    elif "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
    
    model_cfg = cfg.model
    model_type = getattr(model_cfg, "_name", None) # or getattr(cfg, "arch", None)
    
    if model_type == 'wav2vec2':
        model_cfg = merge_with_parent(Wav2Vec2Config(), model_cfg)
        model = Wav2Vec2Model.build_model(model_cfg)
    elif model_type == 'wav2vec_ctc':
        cfg.task['data'] = './' # Set path where dict exists
        task = AudioFinetuningTask.setup_task(cfg.task)
        model_cfg = merge_with_parent(Wav2Vec2CtcConfig(), model_cfg)
        model = Wav2VecCtc.build_model(model_cfg, task)
    
    model = quantization_utils.quantize_model_scalar(model, cfg)

    model.load_state_dict(state['model'], strict=True, model_cfg=cfg.model)

    return model


def load_model_and_config(filename, arg_overrides: Optional[Dict[str, Any]] = None):

    state = load_checkpoint_to_cpu(filename, arg_overrides)

    if "args" in state and state["args"] is not None:
        cfg = convert_namespace_to_omegaconf(state["args"])
    elif "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
    
    model_cfg = cfg.model
    model_type = getattr(model_cfg, "_name", None) # or getattr(cfg, "arch", None)
    task_agnostic = None

    if model_type == 'wav2vec2':
        model_cfg = merge_with_parent(Wav2Vec2Config(), model_cfg)
        model = Wav2Vec2Model.build_model(model_cfg)
        config = state["cfg"]["model"]
        task_agnostic = True
    elif model_type == 'wav2vec_ctc':
        cfg.task['data'] = './' # Set path where dict exists
        task = AudioFinetuningTask.setup_task(cfg.task)
        model_cfg = merge_with_parent(Wav2Vec2CtcConfig(), model_cfg)
        model = Wav2VecCtc.build_model(model_cfg, task)
        config = state["cfg"]["model"]["w2v_args"]["model"]
        task_agnostic = False

    model = quantization_utils.quantize_model_scalar(model, cfg)

    model.load_state_dict(state['model'], strict=True, model_cfg=cfg.model)

    return model, config, task_agnostic