from torch import nn
import torch
import numpy as np
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from dataclasses import dataclass, field

import os, sys
sys.path.insert(0, os.path.dirname(os.getcwd()))

from s3prl.upstream.distiller.module import SplitLinear
from modules.ConvFeatureExtractionModel import ConvFeatureExtractionModelConfig, ConvFeatureExtractionModel
from modules.TransformerSentenceEncoderLayer import TransformerSentenceEncoderLayerConfig, TransformerSentenceEncoderLayer
from modules.DistilTransformerEncoder import TransformerEncoderConfig, DistilTransformerEncoder

@dataclass
class DistillerConfig(FairseqDataclass):
    
    conv_layer_setting: ConvFeatureExtractionModelConfig = field(
        default = ConvFeatureExtractionModelConfig(),
        metadata = {"help": "Default setting of ConvFeatureExtractionModelConfig"}
    )
    # Same config as Distiller version
    encoder_setting: TransformerEncoderConfig = field(
        default = TransformerEncoderConfig(
                    encoder_layers = 1,
                  ),
        metadata = {"help": "Default setting for Distilled model of TransformerEncoderConfig"}
    )
    
    feature_grad_mult: float = field(
        default = 1.0,
        metadata = {"help": "multiply feature extractor var grads by this"}
    )
    
    final_dim: int = field(
        default = 768,
        metadata = {
            "help": "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        },
    ) 
    
    out_layer_type: str = field(
        default = "expand-last",
        metadata = {"help": "None"},
    )
        
    out_layer_inter_dim: int = field(
        default = -1,
        metadata = {"help": "None"},
    )

    n_tasks: int = field(
        default = 3,
        metadata = {"help": "# of loss function between teacher model"},
    )
        
    task_emb_type: str = field(
        default = "expand-last",
        metadata = {"help": "How to define embedding for loss function"},
    )
    
    task_emb_size: int = field(
        default = 0,
        metadata = {"help": "Dimension of corresponding task embedding"},
    ) 
    
    layer_emb_size: int = field(
        default = 0,
        metadata = {"help": "None"},
    )
    
    loss_type: str = field(
        default = "l1",
        metadata = {"help": "Type of loss function for distilling"},
    )
    
    feat_pen_loss: float = field(
        default = 0.0,
        metadata = {"help": "None"},
    )
    
    cosine_loss: float = field(
        default = 0.0,
        metadata = {"help": "Coefficient of cosing loss"},
    )
    
    # When task_emb_type == 'expand-last' only
    pred_layer_id: str = field(
        default = "[3, 7, 11]",
        metadata = {"help": "Index layer for Prediction heads"}
    )
    
    init_teacher_conv_layers: bool = field(
        default = False,
        metadata = {"help": "Initialize to teacher's conv layers"}
    )
    
    init_teacher_encoder_layers: bool = field(
        default = False,
        metadata = {"help": "Initialize to teacher's encoder layers"}
    )
    
class DistillerModel(nn.Module):

    def __init__(self, cfg: DistillerConfig):
        super().__init__()

        # ConFeatureExtraction
        self.conv_layers = eval(cfg.conv_layer_setting.conv_feature_layers)
        feat_emb_dim = self.conv_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(cfg.conv_layer_setting)
        self.feature_grad_mult = cfg.feature_grad_mult

        # Distillation task
        self.n_tasks = cfg.n_tasks
        self.task_emb_type = cfg.task_emb_type
        final_emb_size = cfg.encoder_setting.layer_setting.encoder_embed_dim
        if self.task_emb_type == "add":
            self.task_embedding = nn.Embedding(cfg.n_tasks, cfg.encoder_setting.layer_setting.encoder_embed_dim)
            nn.init.normal_(self.task_embedding.weight, 0.0, 0.1)
        elif self.task_emb_type == "concat":
            assert cfg.task_emb_size > 0
            feat_emb_dim += cfg.task_emb_size
            self.task_embedding = nn.Embedding(cfg.n_tasks, cfg.task_emb_size)
        elif self.task_emb_type == "concat-last":
            assert cfg.task_emb_size > 0
            self.task_embedding = nn.Embedding(cfg.n_tasks, cfg.task_emb_size)
            final_emb_size += cfg.task_emb_size
        elif self.task_emb_type == "expand-last":  ## Default
            self.pred_layer_id = eval(cfg.pred_layer_id)
            assert self.n_tasks == len(self.pred_layer_id)
            print(
                f"[DistillerModel] - Expands the output dimension by {self.n_tasks} times"
            )
            print(f"[DistillerModel] - Pred layers: {self.pred_layer_id}")
        elif self.task_emb_type == "self-hidden":
            self.pred_layer_id = cfg.pred_layer_id
            assert self.n_tasks == len(self.pred_layer_id)
            assert self.n_tasks == cfg.encoder_layers + 1
            print("[DistillerModel] - Predicting with self-hidden layers")
            print(f"[DistillerModel] - Pred layers: {self.pred_layer_id}")
        elif self.task_emb_type == "none":
            print(
                f"[DistillerModel] - Disabled task embedding (predicts only layer {self.n_tasks})"
            )
        else:
            raise NotImplementedError(f"Unknown task emb type {self.task_emb_type}")

        # After ConvFeatureExtraction
        self.post_extract_proj = (
            nn.Linear(feat_emb_dim, cfg.encoder_setting.layer_setting.encoder_embed_dim)
            if feat_emb_dim != cfg.encoder_setting.layer_setting.encoder_embed_dim
            else None
        )

        # TransformerEncoderLayer
        self.encoder_layers = cfg.encoder_setting.encoder_layers
        if cfg.encoder_setting.encoder_layers > 0:
            self.encoder = DistilTransformerEncoder(cfg.encoder_setting)
        else:
            self.encoder = nn.GELU()

        final_dim = cfg.final_dim * (
            1 if self.task_emb_type != "expand-last" else self.n_tasks
        )

        inter_dim = cfg.out_layer_inter_dim
        inter_dim = inter_dim if inter_dim > 0 else final_emb_size

        print(f"[DistillerModel] - Out layer type: {cfg.out_layer_type}")
        if cfg.out_layer_type == "expand-last":
            assert self.task_emb_type == "expand-last"
            print(f"[DistillerModel] - Inter dim = {inter_dim}")
            self.output_layer = nn.Sequential(
                nn.Linear(final_emb_size, inter_dim * self.n_tasks),
                nn.GELU(),
                SplitLinear(inter_dim, self.n_tasks, cfg.final_dim),
            )
        elif cfg.out_layer_type in {"none", "self-hidden"}:
            self.output_layer = None
        else:
            raise NotImplementedError(f"Unknown out layer type {cfg.out_layer_type}")

    def forward_feature(self, wave, pad_mask):
        """Forward feature extractor"""

        if self.feature_grad_mult > 0:
            feat = self.feature_extractor(wave)
            if self.feature_grad_mult != 1.0:
                feat = GradMultiply.apply(feat, self.feature_grad_mult)
        else:
            with torch.no_grad():
                feat = self.feature_extractor(wave)

        feat = feat.transpose(1, 2)  # B x T x D
        pad_mask = self.cal_pad_mask(pad_mask, feat.shape[1])

        return feat, pad_mask

    def forward(self, wave, pad_mask, task_id=None, get_hidden=False, no_pred=False):
        """
        Forward function
        Input:
            wave (FloatTensor): B x T_wave
            pad_mask (BoolTensor): B x T_wave
            task_id (LongTensor): N >= 1
        """

        feat, pad_mask = self.forward_feature(wave, pad_mask)

        if self.task_emb_type not in ["none", "expand-last", "self-hidden"]:
            if task_id is None:
                task_id = self.generate_task_id(feat.device)
            elif isinstance(task_id, list):
                task_id = torch.LongTensor(task_id).to(feat.device)
            task_embs = self.task_embedding(task_id)
            # N x D
            n_sz = len(task_id)
        else:
            n_sz = 1
        b_sz, t_sz, _ = feat.shape

        if self.task_emb_type == "add":
            # Add embs to feature
            if self.post_extract_proj is not None:
                feat_final = self.post_extract_proj(feat)
            else:
                feat_final = feat
            feat_final = feat_final.unsqueeze(1) + task_embs.unsqueeze(0).unsqueeze(2)
        elif self.task_emb_type == "concat":
            # Concatenates embs to feature
            feat_final = torch.cat(
                [
                    feat.unsqueeze(1).expand(-1, n_sz, -1, -1),
                    task_embs.unsqueeze(0).unsqueeze(2).expand(b_sz, -1, t_sz, -1),
                ],
                dim=-1,
            )
            if self.post_extract_proj is not None:
                feat_final = self.post_extract_proj(feat_final)
        else:
            if self.post_extract_proj is not None:
                feat_final = self.post_extract_proj(feat)
            else:
                feat_final = feat
            feat_final = feat_final.unsqueeze(1)
        # feat_final: B x N x T x D or B x 1 x T x D

        pad_mask = pad_mask.unsqueeze(1).expand(-1, n_sz, -1).reshape(b_sz * n_sz, t_sz)
        # BN x T
        feat_final = feat_final.reshape(b_sz * n_sz, t_sz, -1)
        # BN x T x D

        layer_hiddens = []
        if self.encoder_layers > 0:
            get_hidden_tmp = (
                True if (self.task_emb_type == "self-hidden") else get_hidden
            )
            hidden, layer_hiddens = self.encoder(
                feat_final, ~pad_mask.bool(), get_hidden=get_hidden_tmp
            )
        else:
            hidden = self.encoder(feat_final)

        if not no_pred:
            if self.task_emb_type == "self-hidden":
                pred = torch.stack([feat_final] + layer_hiddens, dim=1)
            else:
                pred = self.output_layer(hidden).reshape(b_sz, n_sz, t_sz, -1)
            # B x N x T x D
        else:
            pred = None

        if (not no_pred) and self.task_emb_type == "expand-last":
            assert n_sz == 1, n_sz
            pred = (
                pred.squeeze(1)
                .reshape(b_sz, t_sz, self.n_tasks, -1)
                .permute(0, 2, 1, 3)
            )
            # B x N x T x D

        if get_hidden:
            return feat, feat_final, pred, pad_mask, layer_hiddens
        else:
            return feat, feat_final, pred, pad_mask

    def cal_pad_mask(self, pad_mask, max_len):
        """Calculates pad mask after conv."""
        pad_len = (pad_mask > 0).sum(1).long()
        for _, k_size, s_size in self.conv_layers:
            pad_len = (pad_len - k_size) // s_size + 1

        new_pad_mask = torch.ones(
            (pad_mask.shape[0], max_len), dtype=pad_mask.dtype, device=pad_mask.device
        )

        for idx in range(pad_len.shape[0]):
            new_pad_mask[idx, pad_len[idx] :] = 0

        return new_pad_mask

    def generate_task_id(self, device):
        return torch.arange(self.n_tasks, device=device, dtype=torch.long)