import os
import yaml
import torch
import random
import logging
import numpy as np
from collections import defaultdict
from torch.utils.data import DistributedSampler

from s3prl import hub
from s3prl import downstream
from s3prl.optimizers import get_optimizer
from s3prl.schedulers import get_scheduler
from s3prl.upstream.interfaces import Featurizer

from utils import freeze_model, get_time_tag

from importlib import reload
logging.shutdown()
reload(logging)

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin

# ARGS -------------------------------- <- Some of these should be updated from downstream's config.yaml!!!

# Checkpoint related
OUTPUT_DIR = 'incs_ch-nocopytrf-0.1'
#CHECKPOINT = 'last.ckpt'
CHECKPOINT = None

# MODEL_CHECKPOINT = './results/pretrain/hubert-512D-8H-6L/last.ckpt'
MODEL_CHECKPOINT = './results/pretrain/incs_ch-nocopytrf-0.1/last.ckpt'
# MODEL_CONFIG = './results/pretrain/hubert-512D-8H-6L/2022-02-18_20h09m21s.yaml'
MODEL_CONFIG = './results/pretrain/incs_ch-nocopytrf-0.1/2022-03-07_05h44m17s.yaml'

DOWNSTREAM = 'asr'

UPSTREAM = 'test'
#UPSTREAM = 'distilhubert'
UPSTREAM_TRAINABLE = False
# UPSTREAM_TRAINABLE = True
UPSTREAM_FEATURE_SELECTION = 'last_hidden_state'
# UPSTREAM_FEATURE_SELECTION = 'paper'
UPSTREAM_LAYER_SELECTION = None

# Training related
GPUS = 2
ACCUMULATE_GRAD_BATCHES = 1

SEED = 1339

LIBRI_ROOT = '../db/LibriSpeech'
S3PRL_ROOT = '../s3prl/s3prl'

# Evaluation related
TEST = False
# TEST = True
TEST_SPLIT = 'test-clean' # default: 'test'
# --------------------------------------

# TODO: find out what's the cause of dangling memory references

class W2V2Downstream(LightningModule):
    def __init__(self, 
        args,
        config,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.args = args
        self.config = config

        Upstream = getattr(hub, UPSTREAM)
        self.model = Upstream(
            ckpt=MODEL_CHECKPOINT,
            model_config=MODEL_CONFIG,
        )

        if not UPSTREAM_TRAINABLE:
            freeze_model(self.model)

        self.featurizer = Featurizer(
            upstream=self.model,
            feature_selection=UPSTREAM_FEATURE_SELECTION,
            layer_selection=UPSTREAM_LAYER_SELECTION,
            upstream_device='cpu',
        )

        Downstream = getattr(downstream.experts, DOWNSTREAM)
        self.downstream = Downstream(
            upstream_dim=self.featurizer.output_dim,
            upstream_rate=self.featurizer.downsample_rate,
            expdir=f'results/downstream/{OUTPUT_DIR}',
            **config,
            # **vars(args),
        )

        self.specaug = None
        if self.config.get('specaug'):
            from utils.specaug import SpecAug
            self.specaug = SpecAug(**self.config["specaug"])

        self.train_split = self.config['runner'].get("train_dataloader", "train")
        self.eval_split = self.config['runner']['eval_dataloaders'][0]
        self.test_split = TEST_SPLIT

        self.eval_batch_size = self.config['downstream_expert']['datarc']['eval_batch_size']

        self.train_records = defaultdict(list)
        self.eval_records = defaultdict(list)
        self.test_records = defaultdict(list)

        # For better pytorch lightning logging
        logging.shutdown()
        reload(logging)

    def _get_optimizer(self, model_params):
        optimizer = get_optimizer(
            model_params, 
            self.config['runner']['total_steps'],
            self.config['optimizer']
        )
        return optimizer

    def _get_scheduler(self, optimizer):
        scheduler = get_scheduler(
            optimizer,
            self.config['runner']['total_steps'],
            self.config['scheduler']
        )
        return scheduler

    def configure_optimizers(self):
        if UPSTREAM_TRAINABLE:
            trainable_models = [self.model]
        else:
            self.model.eval()
            trainable_models = []

        trainable_models += [self.featurizer, self.downstream]

        # optimizer
        optimizer = self._get_optimizer(trainable_models)

        # scheduler
        scheduler = None
        if self.config.get('scheduler'):
            scheduler = self._get_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }

        return {
            "optimizer": optimizer,
        }

    def forward(self, wavs):

        wavs = [torch.FloatTensor(wav).to(self.device) for wav in wavs]

        features = self.model(wavs)
        features = self.featurizer(wavs, features)

        return features

    def training_step(self, batch, batch_idx):
        # TODO: find out how s3prl handles deterministic ctc_loss backward pass
        # torch.use_deterministic_algorithms(True)

        (wavs, *others) = batch

        features = self(wavs)

        if self.specaug:
            features, _ = self.specaug(features)

        loss = self.downstream(
            self.train_split,
            features, *others,
            records=self.train_records
        )

        # torch.use_deterministic_algorithms(False)

        return loss

    def training_epoch_end(self, training_step_outputs):
        self.train_records = defaultdict(list)

    def validation_step(self, batch, batch_idx):
        (wavs, *others) = batch

        features = self(wavs)

        self.downstream(
            self.eval_split,
            features, *others,
            records=self.eval_records
        )

    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.FloatTensor(self.eval_records['loss']).mean().item()

        # TODO: generalize to other tasks
        uer, wer = self.downstream._compute_metrics(
            self.eval_records['target_tokens'],
            self.eval_records['target_words'],
            self.eval_records['pred_tokens'],
            self.eval_records['pred_words'],
        )

        self.log("v_loss", loss, on_epoch=True, prog_bar=True, batch_size=self.eval_batch_size)
        self.log("wer", wer, on_epoch=True, prog_bar=True, batch_size=self.eval_batch_size)
        self.log("uer", uer, on_epoch=True, prog_bar=True, batch_size=self.eval_batch_size)

        self.eval_records = defaultdict(list)

    def test_step(self, batch, batch_idx):
        (wavs, *others) = batch

        features = self(wavs)

        self.downstream(
            self.test_split,
            features, *others,
            records=self.test_records
        )
        
    def test_epoch_end(self, test_step_outputs):
        loss = torch.FloatTensor(self.test_records['loss']).mean().item()

        # TODO: generalize to other tasks
        uer, wer = self.downstream._compute_metrics(
            self.test_records['target_tokens'],
            self.test_records['target_words'],
            self.test_records['pred_tokens'],
            self.test_records['pred_words'],
        )

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, batch_size=self.eval_batch_size)
        self.log("wer", wer, on_epoch=True, prog_bar=True, batch_size=self.eval_batch_size)
        self.log("uer", uer, on_epoch=True, prog_bar=True, batch_size=self.eval_batch_size)

        self.test_records = defaultdict(list)

    def train_dataloader(self):
        # epoch = self.init_ckpt.get('Epoch', 0)
        epoch = 0
        try:
            dataloader = self.downstream.get_dataloader(self.train_split, epoch=epoch) # -> What's the purpose of epoch?
            # -> Is there a downstream expert that needs 'epoch' as argument?
        except TypeError as e:
            if "unexpected keyword argument 'epoch'" in str(e):
                dataloader = self.downstream.get_dataloader(self.train_split)
                if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, DistributedSampler):
                    dataloader.sampler.set_epoch(epoch)
            else:
                raise
        
        return dataloader

    def val_dataloader(self):
        dataloader = self.downstream.get_dataloader(self.eval_split)
        return dataloader

    def test_dataloader(self):
        dataloader = self.downstream.get_dataloader(self.test_split)
        return dataloader

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict


if __name__ == '__main__':
    output_dir = f'results/downstream/{OUTPUT_DIR}'
    os.makedirs(output_dir, exist_ok=True)

    # For reproducibility with S3PRL
    seed_everything(SEED, workers=True)

    # TODO: convert all ARGS to args (with parser or yaml)
    args = None

    # Load downstream-specific config from yaml
    config = os.path.join(S3PRL_ROOT, f'downstream/{DOWNSTREAM}/config.yaml')
    with open(config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Update downstream config
    config['downstream_expert']['datarc']['libri_root'] = LIBRI_ROOT
    config['downstream_expert']['datarc']['bucket_file'] = './data/len_for_bucket'
    config['downstream_expert']['datarc']['dict_path'] = S3PRL_ROOT + '/downstream/asr/char.dict'
    config['downstream_expert']['datarc']['train_batch_size'] = (
        config['downstream_expert']['datarc']['train_batch_size'] // ACCUMULATE_GRAD_BATCHES
    )
    config['downstream_expert']['datarc']['batch_size'] = (
        config['downstream_expert']['datarc']['batch_size'] // ACCUMULATE_GRAD_BATCHES
    )

    config['runner']['total_steps'] = (
        config['runner']['total_steps'] // (len(GPUS) if isinstance(GPUS, list) else GPUS)
    )   # -> TODO: must verify this

    # Dump args as yaml file
    if args is not None:
        with open(os.path.join(output_dir, f'args_{get_time_tag()}.yaml'), 'w') as file:
            yaml.dump(vars(args), file)

    # Dump config as yaml file
    with open(os.path.join(output_dir, f'config_{get_time_tag()}.yaml'), 'w') as file:
        yaml.dump(config, file)

    model = W2V2Downstream(
        args=args,
        config=config,
    )

    if CHECKPOINT:
        model = model.load_from_checkpoint(os.path.join(output_dir, CHECKPOINT))

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='checkpoint-{epoch:02d}',
        verbose=True,
        save_last=True,
        save_top_k=1,
        monitor='wer',
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='wer',
        patience=15,
        verbose=True,
        mode='min'
    )

    trainer = Trainer(
        gpus=GPUS,
        strategy=DDPPlugin(find_unused_parameters=False),
        amp_backend="apex",
        amp_level="O2",
        #precision=16,
        max_steps=config['runner']['total_steps'], # -> TODO: must verify this
        sync_batchnorm=True,
        # deterministic=True,   # -> For some reason produces error!
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=config['runner']['gradient_clipping'],
        callbacks=[checkpoint_callback]
        )

    if TEST:
        trainer.test(model)
    else:
        trainer.fit(model)

    