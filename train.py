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

from modules.CustomWav2Vec2Model import CustomWav2Vec2Config, CustomWav2Vec2Model
from utils import *

from importlib import reload
logging.shutdown()
reload(logging)

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# CONFIG ------------------------------
# Teacher model related
TEACHER_MODEL = 'wav2vec2_vox_960h_new.pt'
INIT_TEACHER_CONV_LAYERS = True
INIT_TEACHER_ENCODER_LAYERS = True
N_LAYERS_TO_INIT = 2

# Student model related
STUDENT_ENCODER_LAYERS = 2
ENABLE_TR_LAYER = False
TR_LAYER_FLOOR = 1
TR_TYPE = 'fc2'
TR_REDUCE_FACTOR = 2
PROJ_HEAD_INTER_DIM = 0
PROJ_HEAD_FINAL_DIM = 768

# DB for training related 
DATA_PATH = '../'
DATA_AMOUNT = "100h"
# DATA_AMOUNT = "460h"
# DATA_AMOUNT = "960h"

# Distillation training related
USE_GT_FOR_CTC = True
COSINE_LOSS_WEIGHT = 1
PRED_LAYER_ID = [3, 7, 11]
NUM_EPOCHS = 100
GPUS = 2
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
NUM_WARMUP_STEPS_FACTOR = 0.01
ACCUMULATE_GRAD_BATCHES = 12 # Effective batch size of 24 utterances
MONITOR_LOSSES = True

# Checkpoint related
OUTPUT_DIR = './results/'
# CHECKPOINT = 'last.ckpt'
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

        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Please define configs newly between student and teacher
        self.student_config = self.teacher_config
        self.student_config.encoder_setting.encoder_layers = STUDENT_ENCODER_LAYERS
        self.student_config.encoder_setting.able_tr_layer = ENABLE_TR_LAYER
        self.student_config.encoder_setting.type_of_tr_layer = TR_TYPE
        self.student_config.encoder_setting.tr_layer_floor = TR_LAYER_FLOOR
        self.student_config.proj_head_inter_dim = PROJ_HEAD_INTER_DIM
        self.student_config.proj_head_final_dim = PROJ_HEAD_FINAL_DIM
        self.student_config.final_dropout = 0.0
        self.student_config.targ_d = 32

        # Model Initialize -> Distillation training -> Add FC/Dropout & Fine-tuning
        # self.student_model = CustomWav2Vec2Model(self.student_config)
        self.student_model = CustomStudentModel(self.student_config, self.task_agnostic, PRED_LAYER_ID)

        '''
        if task_agnostic:
            self.student_model = CustomWav2Vec2Model(self.student_config) 
        # Distill model & fine-tuned FC both
        # Use CTC loss or not
        else:
            tmp_cfg = CustomWav2Vec2EncoderConfig()
            tmp_cfg.custom_wav2vec2_model_setting = self.student_model
            self.student_config = tmp_cfg
            self.student_encoder = CustomWav2Vec2Encoder(self.student_config)
            self.student_model = self.student_encoder.custom_model
        '''

        # Copy model parameters of teacher model
        if INIT_TEACHER_CONV_LAYERS:
            self.student_model.init_teacher_conv_layers(self.teacher_model)
        if INIT_TEACHER_ENCODER_LAYERS:
            self.student_model.init_teacher_encoder_layers(self.teacher_model, N_LAYERS_TO_INIT)

        # download & prepare data
        self.train_data = torchaudio.datasets.LIBRISPEECH(DATA_PATH, "train-clean-100", download=True)
        if DATA_AMOUNT == "960h":
            train_data_360 = torchaudio.datasets.LIBRISPEECH(DATA_PATH, "train-clean-360", download=True)
            train_data_500 = torchaudio.datasets.LIBRISPEECH(DATA_PATH, "train-other-500", download=True)
            self.train_data = torch.utils.data.ConcatDataset([self.train_data, train_data_360, train_data_500])
        elif DATA_AMOUNT == "460h":
            train_data_360 = torchaudio.datasets.LIBRISPEECH(DATA_PATH, "train-clean-360", download=True)
            self.train_data = torch.utils.data.ConcatDataset([self.train_data, train_data_360])
        self.eval_data = torchaudio.datasets.LIBRISPEECH(DATA_PATH, "dev-clean", download=True)
        self.test_data = torchaudio.datasets.LIBRISPEECH(DATA_PATH, TEST_SET, download=True)

        # For better pytorch lightning logging
        logging.shutdown()
        reload(logging)

    def forward(self, batch):
        # Seems like lightning had been using the teacher model as training mode the whole time
        self.teacher_model.eval()

        # Input batch into teacher_model
        result = self.teacher_model.w2v_encoder.w2v_model.extract_features(
            source=batch['src'], 
            # padding_mask=batch['mask'],
            padding_mask=None,
            layer=100
        )

        x = result['x'].transpose(0, 1)
        x = self.teacher_model.w2v_encoder.proj(x)

        teacher_results = {
            "encoder_out": x,  # T x B x C
            "padding_mask": result["padding_mask"],  # B x T,
            "layer_results": result["layer_results"],
        }

        # Input batch into student_model
        student_results = self.student_model(batch['src'], padding_mask=None)
        # -> RETURNS: {
        #     "encoder_out": x,  # T x B x C
        #     "padding_mask": result["padding_mask"],  # B x T,
        #     "layer_results": result["layer_results"],
        #     "tr_layer_results": result["tr_layer_results"],
        #     "projections": projections
        # }

        return student_results, teacher_results

    def calculate_loss(self, student_results, teacher_results, labels=None):
        losses = []

        for i, proj in enumerate(student_results['projections']):
            target = teacher_results['layer_results'][PRED_LAYER_ID[i]][0]
        
            rec_loss = F.l1_loss(proj, target)
            
            sim_loss = -F.logsigmoid(F.cosine_similarity(proj, target, dim=-1))
            sim_loss = sim_loss.mean()

            losses.append(rec_loss + COSINE_LOSS_WEIGHT * sim_loss)

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

        return losses

    def training_step(self, batch, batch_idx):
        student_results, teacher_results = self(batch)
        
        if not self.task_agnostic:
            losses = self.calculate_loss(student_results, teacher_results, labels=batch['labels'])
        else:
            losses = self.calculate_loss(student_results, teacher_results)
        loss = sum(losses)

        if MONITOR_LOSSES:
            for i, l in enumerate(losses):
                self.log(f"loss{i+1}", l, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        student_results, teacher_results = self(batch)
        
        if not self.task_agnostic:
            losses = self.calculate_loss(student_results, teacher_results, labels=batch['labels'])
        else:
            losses = self.calculate_loss(student_results, teacher_results)
        loss = sum(losses)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.1, verbose=True)

        train_batches = len(self.train_dataloader()) // GPUS
        num_training_steps = (NUM_EPOCHS * train_batches) // ACCUMULATE_GRAD_BATCHES
        num_warmup_steps = int(num_training_steps * NUM_WARMUP_STEPS_FACTOR)

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
            #     "monitor": "cer",
            # },
        }

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, collate_fn=self.data_collator, num_workers=GPUS*4)

    def val_dataloader(self):
        return DataLoader(self.eval_data, batch_size=self.hparams.batch_size, collate_fn=self.data_collator, num_workers=GPUS*4)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, collate_fn=self.data_collator, num_workers=GPUS*4)

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
        monitor='cer',
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='cer',
        patience=15,
        verbose=True,
        mode='min'
    )

    trainer = Trainer(
        gpus=GPUS,
        strategy="ddp",
        amp_backend="apex",
        amp_level="O2",
        # precision=16, -> Does not work for CTC
        max_epochs=NUM_EPOCHS,
        sync_batchnorm=True,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        callbacks=[early_stopping, checkpoint_callback],
    )

    if TEST:
        trainer.test(model)
    else:
        trainer.fit(model)

    
