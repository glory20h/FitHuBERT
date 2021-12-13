import os
import re
import logging
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from datasets import load_metric

from load_fsq_model import load_model
from modules import ...
from utils import DataCollatorWithPadding

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# CONFIG ------------------------------
TEACHER_MODEL = 'wav2vec2_vox_960h_new.pt'
DATA_PATH = '../'
NUM_EPOCHS = 100
GPUS = 2
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
ACCUMULATE_GRAD_BATCHES = 1
OUTPUT_DIR = './results/'
# CHECKPOINT = 'last.ckpt'
CHECKPOINT = None
# --------------------------------------

class W2V2Distil(LightningModule):
    def __init__(self, 
                model_name=MODEL_NAME,
                data_path=DATA_PATH, 
                learning_rate=LEARNING_RATE,
                batch_size=BATCH_SIZE,
                ):
        super().__init__()

        self.save_hyperparameters()

        self.data_collator = DataCollatorWithPadding(
            feature_extractor=self.feature_extractor, 
            padding=True
        )

        self.wer_metric = load_metric("wer")
        self.cer_metric = load_metric("cer")

        self.teacher_model = load_model(TEACHER_MODEL)
        # Get component of teacher_model
        # self.teacher_model_trnsfrmer_stack = ...

        # Freeze teacher model

        self.student_model = ...

    def forward(self, batch):
        with torch.no_grad():
            # Input batch into teacher_model

        # Input batch into student_model
        # results = ... -> Final(& intermediate) results of student model
        
        # Calculate loss with results
        # loss1 = ...
        # loss2 = ...
        # loss = loss1 + loss2 + ...

        return results, loss

    def training_step(self, batch, batch_idx):
        results, loss = self(batch)

        return loss

    def validation_step(self, batch, batch_idx):
        results, loss = self(batch)

        self.log("v_loss", loss, on_epoch=True, prog_bar=True)
        
        return {"v_loss": loss}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
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
            self.train_data = torchaudio.datasets.LIBRISPEECH(DATA_PATH, download=True)
            self.eval_data = torchaudio.datasets.LIBRISPEECH(DATA_PATH, download=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, collate_fn=self.data_collator, num_workers=GPUS*4)

    def val_dataloader(self):
        return DataLoader(self.eval_data, batch_size=self.hparams.batch_size+1, collate_fn=self.data_collator, num_workers=GPUS*4)

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
        precision=16,
        max_epochs=NUM_EPOCHS,
        sync_batchnorm=True,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        callbacks=[early_stopping, checkpoint_callback],
    )

    if TEST:
        trainer.test(ckpt_path=OUTPUT_DIR + CHECKPOINT)
    else:
        trainer.fit(model)

    