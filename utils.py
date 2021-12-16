import numpy as np
import torch
import torch.nn as nn

from itertools import groupby
from typing import Any, Dict, List

from torch.nn.utils.rnn import pad_sequence

from modules.CustomWav2Vec2 import CustomWav2Vec2Model
from modules.utils_for_asr import Linear

# Alternative for Wav2VecEncoder
class CustomStudentModel(nn.Module):
    def __init__(self, config):
        super(CustomStudentModel, self).__init__()
        self.student_model = CustomWav2Vec2Model(config)
        self.final_dropout = nn.Dropout(config.final_dropout)
        self.proj = Linear(config.encoder_setting.layer_setting.encoder_embed_dim, config.targ_d)
        
    def forward(self, src, padding_mask=None, layer=100):
        result = self.student_model.extract_features(source=src, padding_mask=padding_mask, layer=layer)
        x = result['x'].transpose(0, 1)
        x = self.final_dropout(x)
        x = self.proj(x)
        
        return {
            "encoder_out": x,  # T x B x C
            "padding_mask": result["padding_mask"],  # B x T,
            "layer_results": result["layer_results"],
            "tr_layer_results": result["tr_layer_results"],
        }

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