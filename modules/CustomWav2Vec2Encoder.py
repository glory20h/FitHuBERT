import numpy as np
import torch
import torch.nn as nn

from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass

from modules.CustomWav2Vec2Model import CustomWav2Vec2Config, CustomWav2Vec2Model
from modules.utils import Linear

@dataclass
class CustomWav2Vec2EncoderConfig(FairseqDataclass):
    # Wav2Vec2 Model related
    custom_wav2vec2_model_setting: CustomWav2Vec2Config = field(
        default=CustomWav2Vec2Config(),
        metadata={"help": "config of custom wav2vec2 model"}
    )
    # FC for ASR related
    final_dropout: float = field(  #
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    output_size: int = field(
        default=32,
        metadata={"help": "output dim of fc layer for asr. As same as size of vocab"},
    )

    # time axis masking
    apply_mask: bool = field(  #
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )      

    # Fine-tuning related
    freeze_finetune_updates: int = field(  #
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )

# Alternative for Wav2VecEncoder
class CustomWav2Vec2Encoder(nn.Module):
    def __init__(self, cfg: CustomWav2Vec2EncoderConfig):
        super(CustomWav2Vec2Encoder, self).__init__()
        self.custom_model = CustomWav2Vec2Model(cfg.custom_wav2vec2_model_setting)
        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.proj = Linear(cfg.custom_wav2vec2_model_setting.encoder_setting.layer_setting.encoder_embed_dim, cfg.output_size)
        
    def forward(self, src, padding_mask=None, layer=100):
        result = self.custom_model.extract_features(source=src, padding_mask=padding_mask, layer=layer)
        x = result['x'].transpose(0, 1)
        x = self.final_dropout(x)
        x = self.proj(x)
        
        return {
            "encoder_out": x,  # T x B x C
            "padding_mask": result["padding_mask"],  # B x T,
            "layer_results": result["layer_results"],
            "tr_layer_results": result["tr_layer_results"],
        }
