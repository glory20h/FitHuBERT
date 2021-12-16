import fairseq
from typing import Any, Dict, Optional, Union
from fairseq.data import Dictionary
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.tasks.audio_pretraining import AudioPretrainingTask
from fairseq.tasks.audio_finetuning import AudioFinetuningTask
# from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model, Wav2Vec2Config
from modules.Wav2Vec2Model import Wav2Vec2Model, Wav2Vec2Config
# from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc, Wav2Vec2CtcConfig
from modules.Wav2Vec2Ctc import Wav2VecCtc, Wav2Vec2CtcConfig

from fairseq import models, quantization_utils

from fairseq.dataclass.utils import convert_namespace_to_omegaconf, merge_with_parent

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
    