import os
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from itertools import groupby
from datetime import datetime
from pytz import timezone
from typing import Any, Dict, List, Optional

from fairseq import models, tasks, quantization_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf, merge_with_parent
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.tasks.audio_finetuning import AudioFinetuningTask
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model, Wav2Vec2Config
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc, Wav2Vec2CtcConfig
from fairseq.models.hubert.hubert import HubertModel, HubertConfig
from omegaconf.omegaconf import open_dict


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
        self._hook_layer_hiddens = []
        self._hook_post_cnn = []

        def generate_hook_handler(hiddens: List):
            def hook_handler(self, input, output):
                hiddens.append(output)

            return hook_handler

        self.model.post_extract_proj.register_forward_hook(
                generate_hook_handler(self._hook_post_cnn)
            )

        for module in self.model.encoder.layers:
            module.register_forward_hook(
                generate_hook_handler(self._hook_layer_hiddens) # -> but is it absolutely needed?
            )

    def extract_features(self, source, padding_mask):
        self._hook_layer_hiddens.clear()
        result = {}

        self.model.extract_features(
            source,
            padding_mask,
            mask=None,
        )

        hook_layer_hiddens = self._hook_layer_hiddens.copy()
        self._hook_layer_hiddens.clear()
        hook_post_cnn = self._hook_post_cnn.copy()
        self._hook_post_cnn.clear()

        result['layer_results'] = hook_layer_hiddens
        result['x'] = result['layer_results'][-1][0].transpose(0, 1)
        result['features'] = hook_post_cnn

        return result


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

    dump_dir = './results/pretrain/' + dump_dict['train']['output_dir']
    os.makedirs(dump_dir, exist_ok=True)

    # name as current time
    name = get_time_tag()

    with open(os.path.join(dump_dir, name + '.yaml'), 'w') as f:
        yaml.dump(dump_dict, f, sort_keys = False)
    
    return dump_dict


def get_time_tag():
    return datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d_%Hh%Mm%Ss')


def freeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


def rtrn_attn_forward(
    self,
    x: torch.Tensor,
    self_attn_mask: torch.Tensor = None,
    self_attn_padding_mask: torch.Tensor = None,
    need_weights: bool = True,
    att_args=None,
):
    """
    The substitute forward function for the TransformerSentenceEncoderLayer module.
    It returns unnormalized attention logits which otherwise would not return by default.
    """
    residual = x
    tgt_len, bsz, embed_dim = x.size()

    if self.layer_norm_first:
        x = self.self_attn_layer_norm(x)

        attn_logits, v = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
            before_softmax=True,
        )
        attn_weights_float = F.softmax(
            attn_logits, dim=-1
        )
        attn_weights = attn_weights_float.type_as(attn_logits)
        attn_probs = self.self_attn.dropout_module(attn_weights)
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        x = self.self_attn.out_proj(attn)

        attn = attn_logits
        v_rel = torch.bmm(v * self.self_attn.scaling, v.transpose(1, 2))

        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        layer_result = x

        x = self.dropout3(x)
        x = residual + x
    else:
        attn_logits, v = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            before_softmax=True,
        )
        attn_weights_float = F.softmax(
            attn_logits, dim=-1
        )
        attn_weights = attn_weights_float.type_as(attn_logits)
        attn_probs = self.self_attn.dropout_module(attn_weights)
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        x = self.self_attn.out_proj(attn)

        attn = attn_logits
        v_rel = torch.bmm(v * self.self_attn.scaling, v.transpose(1, 2))

        x = self.dropout1(x)
        x = residual + x

        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        layer_result = x

        x = self.dropout3(x)
        x = residual + x
        x = self.final_layer_norm(x)

    return x, ((attn, v_rel), layer_result)


def con_rtrn_attn_forward(
    self,
    x: torch.Tensor,
    self_attn_mask: torch.Tensor = None,
    self_attn_padding_mask: torch.Tensor = None,
    need_weights: bool = True,
    att_args=None,
    position_emb=None,
):
    """
    Args:
        x: Tensor of shape T X B X C
        self_attn_padding_mask: Optional mask tensor
        positions:
    Returns:
        Tensor of shape T X B X C
    """
    residual = x
    x = self.ffn1(x)
    x = x * 0.5 + residual
    residual = x
    x = self.self_attn_layer_norm(x)
    tgt_len, bsz, embed_dim = x.size()

    if self.pos_enc_type == "rel_pos":
        attn_logits, v = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            pos_emb=position_emb,
            before_softmax=True,
        )
        attn_weights_float = F.softmax(
            attn_logits, dim=-1
        )
        attn_weights = attn_weights_float.type_as(attn_logits)
        attn_probs = self.self_attn.dropout_module(attn_weights)
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        x = self.self_attn.out_proj(attn)

        attn = attn_logits
        v_rel = torch.bmm(v * self.self_attn.scaling, v.transpose(1, 2))
    else:
        attn_logits, v = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            before_softmax=True,
        )
        attn_weights_float = F.softmax(
            attn_logits, dim=-1
        )
        attn_weights = attn_weights_float.type_as(attn_logits)
        attn_probs = self.self_attn.dropout_module(attn_weights)
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        x = self.self_attn.out_proj(attn)

        attn = attn_logits
        v_rel = torch.bmm(v * self.self_attn.scaling, v.transpose(1, 2))
        
    x = self.self_attn_dropout(x)
    x = x + residual

    residual = x
    # TBC to BTC
    x = x.transpose(0, 1)
    x = self.conv_module(x)
    # BTC to TBC
    x = x.transpose(0, 1)
    x = residual + x

    residual = x
    x = self.ffn2(x)

    layer_result = x

    x = x * 0.5 + residual

    x = self.final_layer_norm(x)
    return x, ((attn, v_rel), layer_result)