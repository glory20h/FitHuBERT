from dataclasses import dataclass, field

import torch
import torch.nn as nn

from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel
from fairseq.modules import GradMultiply, LayerNorm

from .module import (
    ConvFeatureExtractionModel,
    TransformerEncoder,
    ConformerEncoder,
    TransformerSentenceEncoderLayer,
    SplitLinear,
)

@dataclass
class CustomStudentModelConfig(FairseqDataclass):

    extractor_mode: str = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group norm with d "
            "groups in the first conv block, whereas layer_norm has layer norms in "
            "every block (meant to use with normalize=True)"
            "Choose from ['default', 'layer_norm']"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: str = field(
        default="gelu",
        metadata={
            "help": "activation function to use"
            "Choose from ['relu', 'gelu', 'gelu_fast', 'gelu_accurate', 'tanh', 'linear']"
        }
    )
    layer_type: str = field(
        default="transformer",
        metadata={
            "help": "layer type in encoder"
            "Choose from ['transformer', 'conformer']"
        }
    )

    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the transformer"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        },
    )
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    feature_grad_mult: float = field(
        default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    pos_conv_depth: int = field(
        default=1,
        metadata={"help": "depth of positional encoder network"},
    )
    max_positions: int = field(default=100000, metadata={"help": "Max positions"})
    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={
            "help": "pad the input to encoder such that the sequence length is divisible by multiple"
        },
    )
    crop_seq_to_multiple: int = field(
        default=1,
        metadata={
            "help": "crop convolutional feature extractor output such that the sequence length is divisible by multiple"
        },
    )

    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "depthwise-conv-kernel-size for convolution in conformer layer"
        },
    )
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})

    # Model Initialization
    init_conv_layers: bool = field(
        default=False,
        metadata={"help": "Whether initialize conv layer of teacher model or not"}
    )

    init_encoder_layers: int = field(
        default=0,
        metadata={"help": "# of layer to initialize encoder layer of teacher model."
                          "For non-positive integer, recognize as False"}
    )

    # Prediction Head
    pred_head_inter_dim: int = field(
        default=0,
        metadata={"help": "Intermediate dimension of prediction head"}
    )

    pred_head_final_dim: int = field(
        default=768,
        metadata={"help": "Final output dimension of prediction head"}
    )

    pred_layer_id: str = field(
        default="[3, 7, 11]",
        metadata={"help": "Layer index to predict by prediction heads"}
    )

    # Time-reduction Layer
    enable_tr_layer: bool = field(
        default=True,
        metadata={"help": "applying time reduction layer or not"}
    )
    
    type_of_tr_layer: str = field(
        default="fc1",
        metadata={"help": "type of time reduction layer"}
    )
    
    tr_conv1d_kernel_stride: str = field(
        default="(2, 2)",
        metadata={"help": "If tr is conv1d, list of kernel and stride for conv1d"}
    )
    
    tr_reduce_factor: int = field(
        default=2,
        metadata={"help": "Factor to reduce time length"}
    )
    
    tr_layer_index: int = field(
        default=1,
        metadata={"help": "In which index should the time reduction layer be inserted"}
    )

    teacher_task_agnostic: bool = field(
        default=True,
        metadata={"help": "Flag to determine whether the teacher model is task-agnostic"}
    )


class CustomStudentModel(BaseFairseqModel):
    def __init__(
        self, 
        cfg: CustomStudentModelConfig,
        teacher_model=None,
    ):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0] # embedding dimension

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.crop_seq_to_multiple = cfg.crop_seq_to_multiple
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.feature_grad_mult = cfg.feature_grad_mult

        encoder_cls = TransformerEncoder
        if cfg.layer_type == "conformer" and cfg.pos_enc_type in ["rel_pos", "rope"]:
            encoder_cls = ConformerEncoder

        self.encoder = encoder_cls(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.init_conv_layers = cfg.init_conv_layers
        self.init_encoder_layers = cfg.init_encoder_layers
        self.teacher_task_agnostic = cfg.teacher_task_agnostic

        if self.init_conv_layers:
            assert teacher_model is not None
            self.init_from_teacher_conv(teacher_model)
        if self.init_encoder_layers > 0:
            assert teacher_model is not None
            self.init_from_teacher_enc(teacher_model, self.init_encoder_layers)

        inter_dim = cfg.pred_head_inter_dim
        pred_head_inter_dim = inter_dim if inter_dim > 0 else cfg.encoder_embed_dim
        pred_head_final_dim = cfg.pred_head_final_dim
        self.pred_layer_id = eval(cfg.pred_layer_id)
        self.n_tasks = len(self.pred_layer_id)

        self.proj_head = nn.Sequential(
            nn.Linear(cfg.encoder_embed_dim, pred_head_inter_dim * self.n_tasks),
            nn.GELU(),
            SplitLinear(pred_head_inter_dim, self.n_tasks, pred_head_final_dim),
        )

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def _disable_projection_heads():
        self.proj_head = None

    def forward(
        self,
        source,
        padding_mask=None,
        layer=None,
    ):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        time_steps_to_drop = features.size(1) % self.crop_seq_to_multiple
        if time_steps_to_drop != 0:
            features = features[:, :-time_steps_to_drop]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-time_steps_to_drop]

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        x, layer_results, tr_layer_results = self.encoder(features, padding_mask=padding_mask, layer=layer)

        if self.proj_head:
            # Get output from projection heads
            b_sz, t_sz, _ = x.shape
            pred = self.proj_head(x).reshape(b_sz, 1, t_sz, -1)
            projections = (
                pred.squeeze(1)
                .reshape(b_sz, t_sz, self.n_tasks, -1)
                .permute(0, 2, 1, 3)
            ) # B x N x T x D
        else:
            projections = None

        return {
            "x": x,
            "padding_mask": padding_mask,
            "features": features,
            "layer_results": layer_results,
            "tr_layer_results": tr_layer_results,
            "projections": projections
        }

    def extract_features(self, source, padding_mask, layer=None):
        res = self.forward(
            source, padding_mask, layer=layer
        )
        return res

    def init_from_teacher_conv(self, teacher_model):
        if not self.teacher_task_agnostic:
            teacher_model = teacher_model.model.w2v_encoder.w2v_model

        self.feature_extractor.load_state_dict(
            teacher_model.model.feature_extractor.state_dict()
        )
        try:
            self.post_extract_proj.load_state_dict(
                teacher_model.model.post_extract_proj.state_dict()
            )
        except:
            pass


    def init_from_teacher_enc(self, teacher_model, n_layers):
        assert n_layers <= self.cfg.encoder_layers

        if not self.teacher_task_agnostic:
            teacher_model = teacher_model.model.w2v_encoder.w2v_model

        self.encoder.pos_conv.load_state_dict(
            teacher_model.model.encoder.pos_conv.state_dict()
        )

        for i in range(n_layers):
            self.encoder.layers[i].load_state_dict(
                teacher_model.model.encoder.layers[i].state_dict()
            )