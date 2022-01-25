import os
import re
import logging
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_metric
from transformers import get_scheduler

from utils.utils import *
from utils.utils_for_custom import *
from utils.dataset import *
from modules.CustomWav2Vec2Model import CustomWav2Vec2Config, CustomWav2Vec2Model

from importlib import reload
logging.shutdown()
reload(logging)

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# CONFIG ------------------------------
# Teacher model related
TEACHER_MODEL = '../parameters/w2v2/wav2vec_small.pt'

# Student model related
STUDENT_ENCODER_LAYERS = 2
ENABLE_TR_LAYER = False
TR_LAYER_FLOOR = 1
TR_TYPE = 'fc2'
TR_REDUCE_FACTOR = 2
PROJ_HEAD_INTER_DIM = 0
PROJ_HEAD_FINAL_DIM = 768

# Agnostic distillation related
INIT_TEACHER_CONV_LAYERS = True
#INIT_TEACHER_ENCODER_LAYERS = True
N_INIT_ENCODER_LAYERS = 2
#COPY_PARAMETERS = True

# Distillation training related
PRED_LAYER_ID = "[3, 7, 11]"
NUM_EPOCHS = 100
COSINE_LOSS_WEIGHT = 1
GPUS = 2
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
WARMUP_PROPORTION = 0
ACCUMULATE_GRAD_BATCHES = 6 # Effective batch size of 24 utterances
MONITOR_LOSSES = True
USE_GT_FOR_CTC = True

# DB for training related 
DATA_PATH = './data/len_for_bucket'
LIBRI_ROOT = '../data/LibriSpeech/'
DATA_SETS = ['train-clean-100']
#DATA_AMOUNT = "100h"

# Checkpoint related
OUTPUT_DIR = './results/'
#CHECKPOINT = 'last.ckpt'
CHECKPOINT = None

# Evaluation related
TEST = False
TEST_SET = "test-clean"
# TEST_SET = "test-other"
# --------------------------------------

class W2V2Distil(LightningModule):
    def __init__(self, 
                data_path=DATA_PATH,
                batch_size=BATCH_SIZE,
                ):
        super().__init__()

        self.save_hyperparameters()

        self.data_collator = DataCollatorWithPadding()

        self.wer_metric = load_metric("wer")
        self.cer_metric = load_metric("cer")
        
        self.decoder = Decoder()
        self.ctc_converter = CTCSequenceConverter(return_type="pt")

        self.char_dict = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, " ": 4, "E": 5, 
            "T": 6, "A": 7, "O": 8, "N": 9, "I": 10, "H": 11, "S": 12, 
            "R": 13, "D": 14, "L": 15, "U": 16, "M": 17, "W": 18, "C": 19, 
            "F": 20, "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25, "K": 26, 
            "'": 27, "X": 28, "J": 29, "Q": 30, "Z": 31}

        self.teacher_model, teacher_config, self.task_agnostic = load_model_and_config(TEACHER_MODEL)
        self.teacher_config = convert_dict_to_custom_config(teacher_config)

        freeze_model(self.teacher_model)

        # Assign more configs about student model compares to teacher model
        self.student_config = self.teacher_config        
        self.set_student_config()

        # Model Initialize -> Distillation training -> Add FC/Dropout & Fine-tuning
        self.student_model = CustomWav2Vec2Model(cfg = self.student_config,
                                                 teacher_model = self.teacher_model)

        # download & prepare data
        self.train_data = LibriDataset(
            batch_size=BATCH_SIZE,
            file_path=DATA_PATH,
            sets=DATA_SETS,
            libri_root=LIBRI_ROOT,
        )
        self.eval_data = LibriDataset(
            batch_size=BATCH_SIZE,
            file_path=DATA_PATH,
            sets=['dev-clean'],
            libri_root=LIBRI_ROOT,
        )
        self.test_data = torchaudio.datasets.LIBRISPEECH('../data', TEST_SET, download=True)

        # For better pytorch lightning logging
        logging.shutdown()
        reload(logging)

    def forward(self, batch):
        wave_input, wave_orig, wave_len, pad_mask = batch

        wav_lengths = torch.LongTensor([len(wav) for wav in wave_orig])
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wave_orig, batch_first=True)

        # Seems like lightning had been using the teacher model as training mode the whole time
        self.teacher_model.eval()

        teacher_results = self.teacher_model.extract_features(
            source=padded_wav, 
            padding_mask=wav_padding_mask,
            mask = False,
            layer=100
        )
        # -> RETURNS: {
        #     "x": (B x T x D) (encoder output),
        #     "padding_mask": (B x T),
        #     "features": (B x T x D_conv) (CNN output),
        #     "layer_results": [((T x B x D),(B x T x T))] x 12,
        # }

        student_results = self.student_model.extract_features(
            source=padded_wav, 
            padding_mask=wav_padding_mask,
            mask=False,
            layer=100
        )
        # -> RETURNS: {
        #     "encoder_out": x,  # T x B x C
        #     "padding_mask": result["padding_mask"],  # B x T,
        #     "layer_results": result["layer_results"],
        #     "tr_layer_results": result["tr_layer_results"],
        #     "projections": projections
        # }

        return student_results, teacher_results

    def calculate_loss(self, student_results, teacher_results, labels=None):

        teacher_hiddens = [
            teacher_results["layer_results"][i][0].transpose(0, 1)
            for i in self.student_model.pred_layer_id
        ]
        
        teacher_hiddens = torch.stack(teacher_hiddens, dim=1)  # B x N x T x D
        
        proj = student_results['projections']
        target = teacher_hiddens
        
        rec_loss = F.l1_loss(proj, target, reduction="none")
        with torch.no_grad():
            rec_layer_loss = rec_loss.mean((0, 2, 3))
            
        rec_loss = rec_loss.mean()
        
        if COSINE_LOSS_WEIGHT > 0:
            sim_loss = -F.logsigmoid(F.cosine_similarity(proj, target, dim=-1))
            with torch.no_grad():
                sim_layer_loss = sim_loss.mean((0, 2))
            sim_loss = sim_loss.mean()
        else:
            sim_loss = 0
            sim_layer_loss = None
            
        total_loss = rec_loss + COSINE_LOSS_WEIGHT * sim_loss
        
        losses = torch.add(rec_layer_loss, sim_layer_loss)
        
        return total_loss, losses

        if not self.task_agnostic:
            # Process output for CTC loss
            ctc_input = student_results['encoder_out'].log_softmax(2) # -> Revise this

            if USE_GT_FOR_CTC:
                # Use Ground Truth labels instead of labels from the teacher model
                gt_tokens = [torch.tensor([self.char_dict[char] for char in label]) for label in labels]
                target = torch.cat(gt_tokens)
                target_lengths = torch.tensor([len(tokens) for tokens in gt_tokens])
            else:
                logits = teacher_results['encoder_out'].transpose(0,1)
                predicted_ids = torch.argmax(logits, dim=-1)
                fused_tokens = [self.ctc_converter(ids) for ids in predicted_ids]
                target = torch.cat(fused_tokens)
                target_lengths = torch.tensor([len(tokens) for tokens in fused_tokens])

            ctc_loss = F.ctc_loss(
                ctc_input, 
                target, 
                torch.full((ctc_input.shape[1],), ctc_input.shape[0]),
                target_lengths
            )

            losses.append(ctc_loss)

        return total_loss, losses


    def set_student_config(self):
        # Set student w2v model configs before distillation

        # Model spec realted
        self.student_config.encoder_setting.encoder_layers = STUDENT_ENCODER_LAYERS
        self.student_config.encoder_setting.able_tr_layer = ENABLE_TR_LAYER
        self.student_config.encoder_setting.type_of_tr_layer = TR_TYPE
        self.student_config.encoder_setting.tr_layer_floor = TR_LAYER_FLOOR
        
        # Initialization related
        self.student_config.init_conv_layers = INIT_TEACHER_CONV_LAYERS
        self.student_config.init_encoder_layers = N_INIT_ENCODER_LAYERS

        # Prediction head related
        self.student_config.proj_head_inter_dim = PROJ_HEAD_INTER_DIM
        self.student_config.proj_head_final_dim = PROJ_HEAD_FINAL_DIM
        self.student_config.pred_layer_id = PRED_LAYER_ID
        self.student_config.teacher_task_agnostic = self.task_agnostic


    def training_step(self, batch, batch_idx):
        student_results, teacher_results = self(batch)
        
        if not self.task_agnostic:
            loss, losses = self.calculate_loss(student_results, teacher_results, labels=batch['labels'])
        else:
            loss, losses = self.calculate_loss(student_results, teacher_results)

        if MONITOR_LOSSES:
            for i, l in enumerate(losses):
                self.log(f"loss{i+1}", l.item(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        student_results, teacher_results = self(batch)

        if not self.task_agnostic:
            loss, losses = self.calculate_loss(student_results, teacher_results, labels=batch['labels'])
        else:
            loss, losses = self.calculate_loss(student_results, teacher_results)

        if not self.task_agnostic:
            predicted_ids = np.argmax(student_results['encoder_out'].transpose(0,1).cpu().detach().numpy(), axis=-1)
            predictions = [self.decoder.decode(ids) for ids in predicted_ids]

            self.wer_metric.add_batch(predictions=predictions, references=batch['labels'])
            self.cer_metric.add_batch(predictions=predictions, references=batch['labels'])

        self.log("v_loss", loss, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)

        return {"v_loss": loss}

    def validation_epoch_end(self, validation_step_outputs):
        if not self.task_agnostic:
            wer = self.wer_metric.compute()
            cer = self.cer_metric.compute()

            self.log("wer", wer, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
            self.log("cer", cer, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
    
    def test_step(self, batch, batch_idx):
        student_results, teacher_results = self(batch)
        
        if not self.task_agnostic:
            losses = self.calculate_loss(student_results, teacher_results, labels=batch['labels'])
        else:
            losses = self.calculate_loss(student_results, teacher_results)
        loss = sum(losses)

        if not self.task_agnostic:
            predicted_ids = np.argmax(student_results['encoder_out'].transpose(0,1).cpu().detach().numpy(), axis=-1)
            predictions = [self.decoder.decode(ids) for ids in predicted_ids]

            wer = self.wer_metric.add_batch(predictions=predictions, references=batch['labels'])
            cer = self.cer_metric.add_batch(predictions=predictions, references=batch['labels'])

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)

        return {"test_loss": loss}

    def test_epoch_end(self, test_step_outputs):
        if not self.task_agnostic:
            wer = self.wer_metric.compute()
            cer = self.cer_metric.compute()

            self.log("wer", wer, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
            self.log("cer", cer, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.1, verbose=True)

        train_batches = len(self.train_dataloader()) // GPUS
        num_training_steps = (NUM_EPOCHS * train_batches) // ACCUMULATE_GRAD_BATCHES
        num_warmup_steps = int(num_training_steps * WARMUP_PROPORTION)

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": lr_scheduler,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "monitor": "v_loss",
            # },
        }

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=1, shuffle=True, collate_fn=self.train_data.collate_fn, num_workers=GPUS*4)

    def val_dataloader(self):
        return DataLoader(self.eval_data, batch_size=1, collate_fn=self.eval_data.collate_fn, num_workers=GPUS*4)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, collate_fn=self.data_collator, num_workers=GPUS*4)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict

if __name__ == '__main__':
    if CHECKPOINT:
        model = W2V2Distil().load_from_checkpoint(OUTPUT_DIR + CHECKPOINT)
    else:
        model = W2V2Distil()

    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename='checkpoint-{epoch:02d}',
        verbose=True,
        save_last=True,
        save_top_k=3,
        monitor='v_loss',
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='v_loss',
        patience=15,
        verbose=True,
        mode='min'
    )

    trainer = Trainer(
        gpus=GPUS,
        strategy="ddp",
        amp_backend="apex",
        amp_level="O2",
        precision=16,
        max_epochs=NUM_EPOCHS,
        sync_batchnorm=True,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        callbacks=[early_stopping, checkpoint_callback],
    )

    if TEST:
        trainer.test(model)
    else:
        trainer.fit(model)

    
