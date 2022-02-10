import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import math

from itertools import groupby
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from omegaconf import MISSING, II, open_dict

from argparse import Namespace
import contextlib
import copy

from fairseq import models, tasks, quantization_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf, merge_with_parent
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.tasks.audio_finetuning import AudioFinetuningTask
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model, Wav2Vec2Config
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc, Wav2Vec2CtcConfig
from fairseq.models.hubert.hubert import HubertModel, HubertConfig
from omegaconf.omegaconf import open_dict

class DataCollatorWithPadding:
    def __call__(self, features: List[Dict[str, Any]]):
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [feature[0][0] for feature in features]
        labels = [feature[2] for feature in features]
        
        src = pad_sequence(input_features, batch_first=True, padding_value=0)
        mask = torch.zeros(src.shape).masked_fill_(src==0, 1)
        
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


class TeacherWrapper(nn.Module):
    def __init__(
        self,
        model,
    ):
        """
        Wrapper for the teacher model 
        The wrapper makes it possible to get every intermediate outputs via hooks
        """
        super().__init__()
        self.model = model
        self._hook_hiddens = []

        def generate_hook_handler(hiddens: List):
            def hook_handler(self, input, output):
                hiddens.append(output)

            return hook_handler

        for module in self.model.encoder.layers:
            module.register_forward_hook(
                generate_hook_handler(self._hook_hiddens) # -> but is it absolutely needed?
            )

    def extract_features(self, source, padding_mask):
        self._hook_hiddens.clear()
        result = {}

        self.model.extract_features(
            source,
            padding_mask,
            mask=None,
        )

        hook_hiddens = self._hook_hiddens.copy()
        self._hook_hiddens.clear()

        result['layer_results'] = hook_hiddens
        result['x'] = result['layer_results'][-1][0].transpose(0, 1)

        return result


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
        with open_dict(model_cfg):
            model_cfg.required_seq_len_multiple = 1
        model = Wav2Vec2Model.build_model(model_cfg)
        task_agnostic = True
    elif model_type == 'wav2vec_ctc':
        cfg.task['data'] = './' # Set path where dict exists
        task = AudioFinetuningTask.setup_task(cfg.task)
        model_cfg = merge_with_parent(Wav2Vec2CtcConfig(), model_cfg)
        with open_dict(model_cfg):
            model_cfg.required_seq_len_multiple = 1
        model = Wav2VecCtc.build_model(model_cfg, task)
        task_agnostic = False
    elif model_type == "hubert":
        task = tasks.setup_task(cfg.task)
        task.load_state_dict(state["task_state"])
        model_cfg = merge_with_parent(HubertConfig(), model_cfg)
        # Update needed due to a bug in latest version of fairseq
        with open_dict(model_cfg):
            model_cfg.required_seq_len_multiple = 1
            model_cfg.layer_type = 'transformer'
        model = HubertModel.build_model(model_cfg, task)
        task_agnostic = True
    else:
        raise NotImplementedError(f"model '{model_type}' is not supported.")

    model = quantization_utils.quantize_model_scalar(model, cfg)
    model.load_state_dict(state['model'], strict=True, model_cfg=cfg.model)

    # Wrap Teacher
    model.encoder.layerdrop = 0
    model = TeacherWrapper(model)

    return model, model_cfg, task_agnostic

def freeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False
