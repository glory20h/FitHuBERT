import os
import re
import logging
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from datasets import load_metric

from load_fsq_model import load_model
from modules.CustomWav2Vec2 import CustomWav2Vec2Config
from utils import *

from importlib import reload

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# CONFIG ------------------------------
TEACHER_MODEL = 'wav2vec2_vox_960h_new.pt'
DATA_PATH = '../'
STUDENT_ENCODER_LAYERS = 6
TR_LAYER_FLOOR = 3
TR_TYPE = "conv1d"
NUM_EPOCHS = 100
GPUS = 2
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
ACCUMULATE_GRAD_BATCHES = 1
OUTPUT_DIR = './results/'
# CHECKPOINT = 'last.ckpt'
CHECKPOINT = None
TEST = False
# --------------------------------------

class W2V2Distil(LightningModule):
    def __init__(self, 
                data_path=DATA_PATH, 
                learning_rate=LEARNING_RATE,
                batch_size=BATCH_SIZE,
                ):
        super().__init__()

        self.save_hyperparameters()

        self.data_collator = DataCollatorWithPadding()

        self.wer_metric = load_metric("wer")
        self.cer_metric = load_metric("cer")
        
        self.L1loss = nn.L1Loss()
        self.CTCloss = nn.CTCLoss(blank=4, zero_infinity=False) # -> Exp zero_infinity
        
        self.decoder = Decoder()
        self.ctc_converter = CTCSequenceConverter(return_type="pt")

        self.teacher_model = load_model(TEACHER_MODEL)
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Get components of teacher_model
        self.teacher_tf_encoder = self.teacher_model.w2v_encoder.w2v_model.encoder.layers # -> TransformerSentenceEncoder stacks

        self.student_config = CustomWav2Vec2Config()

        # Update student model config as required
        self.student_config.conv_layer_setting.extractor_mode = "layer_norm"
        self.student_config.conv_bias = True
        self.student_config.encoder_setting.layer_setting.encoder_embed_dim = 1024
        self.student_config.encoder_setting.layer_setting.encoder_ffn_embed_dim = 4096
        self.student_config.encoder_setting.layer_setting.encoder_attention_heads = 16
        self.student_config.encoder_setting.layer_setting.dropout = 0.0
        self.student_config.encoder_setting.layer_setting.layer_norm_first=True
        self.student_config.encoder_setting.type_of_tr_layer = TR_TYPE
        self.student_config.encoder_setting.encoder_layers = STUDENT_ENCODER_LAYERS
        self.student_config.encoder_setting.tr_layer_floor = TR_LAYER_FLOOR
        self.student_config.encoder_setting.dropout_input = 0.1
        self.student_config.encoder_setting.dropout_features = 0.1
        self.student_config.encoder_setting.final_dim = 768
        self.student_config.encoder_setting.latent_temp = (2, 0.1, 0.999995)
        self.student_config.final_dropout = 0.0
        self.student_config.targ_d = 32

        self.student_model = CustomStudentModel(self.student_config)

        # download
        self.train_data = torchaudio.datasets.LIBRISPEECH(DATA_PATH, "train-clean-100", download=True) # -> Must use all 960h later
        self.eval_data = torchaudio.datasets.LIBRISPEECH(DATA_PATH, "dev-clean", download=True)
        self.test_data = torchaudio.datasets.LIBRISPEECH(DATA_PATH, "test-clean", download=True)

        # For better pytorch lightning logging
        logging.shutdown()
        reload(logging)

    def forward(self, batch):
        # Input batch into teacher_model
        result = self.teacher_model.w2v_encoder.w2v_model.extract_features(
            source=batch['src'], 
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
        # -> {
        #     "encoder_out": x,  # T x B x C
        #     "padding_mask": result["padding_mask"],  # B x T,
        #     "layer_results": result["layer_results"],
        #     "tr_layer_results": result["tr_layer_results"],
        # }

        
        # Input intermediate result of student model into upper transformer layers of teacher model
        x = student_results['tr_layer_results'][0].detach()

        ### MUST REVISE ###
        for i, layer in enumerate(self.teacher_tf_encoder):
            if i >= 12:
                x, z = layer(x)

        x = x.transpose(0, 1)
        if self.teacher_model.w2v_encoder.w2v_model.encoder.layer_norm_first:
            x = self.teacher_model.w2v_encoder.w2v_model.encoder.layer_norm(x)
        x = x.transpose(0, 1)
        teacher_tf_encoder_out = self.teacher_model.w2v_encoder.proj(x)
        ### MUST REVISE ###
        
        # Process output for CTC loss
        ctc_input = student_results['encoder_out'].log_softmax(2) # -> Revise this
        logits = teacher_results['encoder_out'].transpose(0,1) # T x B x C -> B x T x C
        predicted_ids = torch.argmax(logits, dim=-1)
        fused_tokens = [self.ctc_converter(ids) for ids in predicted_ids]
        target = torch.cat(fused_tokens)
        target_lengths = torch.tensor([len(tokens) for tokens in fused_tokens]) # -> Revise this
        
        # Calculate loss with results
        loss1 = self.L1loss(
                student_results['layer_results'][TR_LAYER_FLOOR-1][0], 
                teacher_results['layer_results'][len(self.teacher_tf_encoder)//2-1][0]
            )
        loss2 = self.L1loss(student_results['encoder_out'], teacher_tf_encoder_out)
        loss3 = self.CTCloss(
                ctc_input, 
                target, 
                torch.full(size=(ctc_input.shape[1],), fill_value=ctc_input.shape[0]), # -> Revise this
                target_lengths
            )
        
        # Can also try weighted sum
        loss = loss1 + loss2 + loss3

        return student_results, loss

    def training_step(self, batch, batch_idx):
        results, loss = self(batch)

        return loss

    def validation_step(self, batch, batch_idx):
        results, loss = self(batch)

        predicted_ids = np.argmax(results['encoder_out'].transpose(0,1).cpu().detach().numpy(), axis=-1)
        predictions = [self.decoder.decode(ids) for ids in predicted_ids]

        wer = self.wer_metric.compute(predictions=predictions, references=batch['labels'])
        cer = self.cer_metric.compute(predictions=predictions, references=batch['labels'])

        self.log("v_loss", loss, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log("wer", wer, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log("cer", cer, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        
        return {"v_loss": loss, "wer": wer, "cer": cer}
    
    def test_step(self, batch, batch_idx):
        results, loss = self(batch)

        predicted_ids = np.argmax(results['encoder_out'].transpose(0,1).cpu().detach().numpy(), axis=-1)
        predictions = [self.decoder.decode(ids) for ids in predicted_ids]

        wer = self.wer_metric.compute(predictions=predictions, references=batch['labels'])
        cer = self.cer_metric.compute(predictions=predictions, references=batch['labels'])

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log("wer", wer, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log("cer", cer, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        
        return {"test_loss": loss, "wer": wer, "cer": cer}
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.1, verbose=True)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "v_loss",
            },
        }

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = torchaudio.datasets.LIBRISPEECH(DATA_PATH, "train-clean-100", download=True) # -> Must use all 960h later
            self.eval_data = torchaudio.datasets.LIBRISPEECH(DATA_PATH, "dev-clean", download=True)
        if stage == "test" or stage is None:
            self.test_data = torchaudio.datasets.LIBRISPEECH(DATA_PATH, "test-clean", download=True)

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
        model = W2V2Distil.load_from_checkpoint(OUTPUT_DIR + CHECKPOINT)
    else:
        model = W2V2Distil()

    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename='checkpoint-{epoch:02d}',
        verbose=True,
        save_last=True,
        save_top_k=3,
        monitor="v_loss",
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
        # precision=16, -> Does not work for CTC
        max_epochs=NUM_EPOCHS,
        sync_batchnorm=True,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        callbacks=[early_stopping, checkpoint_callback],
    )

    if TEST:
        trainer.test(ckpt_path=OUTPUT_DIR + CHECKPOINT)
    else:
        trainer.fit(model)

    