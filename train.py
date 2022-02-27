import os
import re
import yaml
import argparse
import logging
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from s3prl.optimizers import get_optimizer

from utils import *
from modules.model import CustomStudentModelConfig, CustomStudentModel

from importlib import reload
logging.shutdown()
reload(logging)

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class W2V2Distil(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.yaml_cfg = cfg
        self.train_cfg = cfg['train']

        # Load teacher model
        teacher_model = self.yaml_cfg['teacher']['teacher_model']
        self.teacher_model, teacher_config, self.task_agnostic = load_model_and_config(teacher_model)
        freeze_model(self.teacher_model)

        # Make student config independent of teacher
        distiller_cfg = self.yaml_cfg['distiller']
        student_config = CustomStudentModelConfig(**distiller_cfg)
        student_config.teacher_task_agnostic = self.task_agnostic

        # TODO: how to make it save only once?
        # if self.trainer.is_global_zero:
        dump_yaml(student_config, self.yaml_cfg)

        # Model Initialize -> Distillation training -> Add FC/Dropout & Fine-tuning
        self.student_model = CustomStudentModel(
            cfg=student_config,
            teacher_model=self.teacher_model
        )

        self.rec_loss_weight = self.train_cfg['rec_loss_weight']
        self.rec_loss_type = self.train_cfg['rec_loss_type']
        self.sim_loss_weight = self.train_cfg['sim_loss_weight']
        self.attn_loss_weight = self.train_cfg['attn_loss_weight']
        self.attn_loss_type = self.train_cfg['attn_loss_type']
        self.v_rel_loss_weight = self.train_cfg['v_rel_loss_weight']

        if self.attn_loss_weight > 0:
            for layer in self.teacher_model.model.encoder.layers:
                layer.self_attn._set_skip_embed_dim_check()
                bound_method = rtrn_attn_forward.__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)
            for layer in self.student_model.encoder.layers:
                if self.yaml_cfg['distiller']['layer_type'] == 'conformer':
                    layer.self_attn._set_skip_embed_dim_check()
                    bound_method = con_rtrn_attn_forward.__get__(layer, layer.__class__)
                    setattr(layer, 'forward', bound_method)
                else:
                    layer.self_attn._set_skip_embed_dim_check()
                    bound_method = rtrn_attn_forward.__get__(layer, layer.__class__)
                    setattr(layer, 'forward', bound_method)

        if self.train_cfg['no_projections']:
            self.student_model._disable_projection_heads()

        self.batch_size = self.train_cfg['batch_size']
        self.num_gpus = self.train_cfg['gpus']
        if isinstance(self.num_gpus, list):
            self.num_gpus = len(self.num_gpus)
        data_cfg = self.yaml_cfg['data']
        bucketing_path = data_cfg['bucketing_path']
        libri_root = data_cfg['libri_root']
        train_set = data_cfg['train_set']
        test_set = data_cfg['test_set']

        # download & prepare data
        self.train_data = LibriDataset(
            batch_size=self.batch_size,
            file_path=bucketing_path,
            sets=train_set,
            libri_root=libri_root,
        )
        self.eval_data = LibriDataset(
            batch_size=self.batch_size,
            file_path=bucketing_path,
            sets=['dev-clean'],
            libri_root=libri_root,
        )
        self.test_data = LibriDataset(
            batch_size=self.batch_size,
            file_path=bucketing_path,
            sets=test_set,
            libri_root=libri_root,
        )

        # For better pytorch lightning logging
        logging.shutdown()
        reload(logging)

    def forward(self, x, padding_mask=None):
        # Seems like lightning had been using the teacher model as training mode the whole time
        self.teacher_model.eval()

        teacher_results = self.teacher_model.extract_features(
            source=x, 
            padding_mask=padding_mask,
        )
        # -> RETURNS: {
        #     "x": (B x T x D) (encoder output),
        #     "layer_results": [x, (attn, lr)] x #layers,
        # }

        student_results = self.student_model(
            source=x, 
            padding_mask=padding_mask,
        )
        # -> RETURNS: {
        #     "x": x,
        #     "padding_mask": padding_mask,
        #     "features": features,
        #     "layer_results": layer_results,
        #     "tr_layer_results": tr_layer_results,
        #     "projections": projections
        # }

        return student_results, teacher_results

    def training_step(self, batch, batch_idx):
        student_results, teacher_results = self(**batch)
        
        if not self.task_agnostic:
            loss, losses = self.calculate_loss(student_results, teacher_results, labels=batch['labels'])
        else:
            loss, losses = self.calculate_loss(student_results, teacher_results)

        if self.train_cfg['monitor_losses']:
            for k, v in losses.items():
                self.log(k, v.item(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        student_results, teacher_results = self(**batch)

        if not self.task_agnostic:
            loss, losses = self.calculate_loss(student_results, teacher_results, labels=batch['labels'])
        else:
            loss, losses = self.calculate_loss(student_results, teacher_results)

        if not self.task_agnostic:
            predicted_ids = np.argmax(student_results['encoder_out'].transpose(0,1).cpu().detach().numpy(), axis=-1)
            predictions = [self.decoder.decode(ids) for ids in predicted_ids]

            self.wer_metric.add_batch(predictions=predictions, references=batch['labels'])
            self.cer_metric.add_batch(predictions=predictions, references=batch['labels'])

        self.log("v_loss", loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return {"v_loss": loss}

    def validation_epoch_end(self, validation_step_outputs):
        if not self.task_agnostic:
            wer = self.wer_metric.compute()
            cer = self.cer_metric.compute()

            self.log("wer", wer, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
            self.log("cer", cer, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
    
    def test_step(self, batch, batch_idx):
        student_results, teacher_results = self(**batch)
        
        if not self.task_agnostic:
            loss, losses = self.calculate_loss(student_results, teacher_results, labels=batch['labels'])
        else:
            loss, losses = self.calculate_loss(student_results, teacher_results)

        if not self.task_agnostic:
            predicted_ids = np.argmax(student_results['encoder_out'].transpose(0,1).cpu().detach().numpy(), axis=-1)
            predictions = [self.decoder.decode(ids) for ids in predicted_ids]

            wer = self.wer_metric.add_batch(predictions=predictions, references=batch['labels'])
            cer = self.cer_metric.add_batch(predictions=predictions, references=batch['labels'])

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return {"test_loss": loss}

    def test_epoch_end(self, test_step_outputs):
        if not self.task_agnostic:
            wer = self.wer_metric.compute()
            cer = self.cer_metric.compute()

            self.log("wer", wer, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
            self.log("cer", cer, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

    def calculate_loss(self, student_results, teacher_results, labels=None):
        # TODO: move calculate_loss to utils?
        losses = {}
    
        # Feature loss
        if self.rec_loss_weight > 0:
            teacher_hiddens = [
                teacher_results["layer_results"][i][0].transpose(0, 1)
                for i in self.student_model.pred_layer_id
            ]
            
            teacher_hiddens = torch.stack(teacher_hiddens, dim=1)  # B x N x T x D
            
            if self.train_cfg['no_projections']:
                pred = torch.stack([
                    student_results["layer_results"][-1][0].transpose(0, 1)
                    for i in self.student_model.pred_layer_id
                ], dim=1)
            else:
                pred = student_results['projections']
            target = teacher_hiddens.narrow(2, 0, pred.shape[2])
            
            if rec_loss_type == 'l1':
                rec_loss = F.l1_loss(pred, target, reduction="none")
            elif rec_loss_type == 'mse':
                rec_loss = F.mse_loss(pred, target, reduction="none")
            else:
                raise NotImplementedError("rec_loss_type must be one of 'l1', 'mse'.")
            with torch.no_grad():
                rec_layer_loss = rec_loss.mean((0, 2, 3))
                
            rec_loss = rec_loss.mean()
        else:
            rec_loss = 0
            rec_layer_loss = 0
        
        if self.sim_loss_weight > 0:
            sim_loss = -F.logsigmoid(F.cosine_similarity(pred, target, dim=-1))
            with torch.no_grad():
                sim_layer_loss = sim_loss.mean((0, 2))
            sim_loss = sim_loss.mean()
        else:
            sim_loss = 0
            sim_layer_loss = 0

        feat_loss = torch.add(rec_layer_loss, sim_layer_loss)
        
        for i, pred_id in enumerate(self.student_model.pred_layer_id):
            losses[f'layer{pred_id}'] = feat_loss[i]

        # Attention distribution transfer loss
        if self.attn_loss_weight > 0:
            pred = student_results['layer_results'][-1][1][0]
            target = teacher_results['layer_results'][-1][1][0][0]

            if self.attn_loss_type == 'mse':
                loss = F.mse_loss(
                    pred,
                    target,
                    reduction='none'
                )
                inf_count = torch.any(loss.isinf(), 1).count_nonzero() * loss.size(-1)
                nan_count = torch.any(loss.isnan(), 1).count_nonzero() * loss.size(-1)
                loss[loss.isinf()] = 0
                loss[loss.isnan()] = 0
                attn_loss = loss.sum() / (loss.numel() - inf_count - nan_count)
            elif self.attn_loss_type == 'kldiv':
                loss = F.kl_div(
                    F.log_softmax(pred, dim=-1), 
                    F.softmax(target, dim=-1), 
                    reduction='none',
                )
                loss[loss.isinf()] = 0
                attn_loss = loss.sum(dim=-1).mean()
            else:
                raise NotImplementedError("attn_loss_type must be one of 'mse', 'kldiv'.")

            losses['attn_loss'] = attn_loss
        else:
            attn_loss = 0

        # Value Relation Transfer Loss
        if self.v_rel_loss_weight > 0:
            pred = student_results['layer_results'][-1][1][1]
            target = teacher_results['layer_results'][-1][1][0][1]
            loss = F.kl_div(
                F.log_softmax(pred, dim=-1), 
                F.softmax(target, dim=-1), 
                reduction='none',
            )
            v_rel_loss = loss.sum(dim=-1).mean()
            
            losses['v_rel_loss'] = v_rel_loss
        else:
            v_rel_loss = 0
            
        total_loss = (
            self.rec_loss_weight * rec_loss
            + self.sim_loss_weight * sim_loss 
            + self.attn_loss_weight * attn_loss
            + self.v_rel_loss_weight * v_rel_loss
        )

        if not self.task_agnostic:
            # Process output for CTC loss
            ctc_input = student_results['x'].log_softmax(2) # -> Revise this

            if self.train_cfg['use_gt_for_ctc']:
                # Use Ground Truth labels instead of labels from the teacher model
                gt_tokens = [torch.tensor([self.char_dict[char] for char in label]) for label in labels]
                target = torch.cat(gt_tokens)
                target_lengths = torch.tensor([len(tokens) for tokens in gt_tokens])
            else:
                logits = teacher_results['x'].transpose(0,1)
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

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=eval(self.yaml_cfg['optimizer']['lr']))
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.1, verbose=True)

        train_batches = len(self.train_dataloader()) // self.num_gpus
        num_training_steps = (self.train_cfg['num_epochs'] * train_batches) // self.train_cfg['accumulate_grad_batches']
        num_warmup_steps = int(num_training_steps * self.yaml_cfg['optimizer']['warmup_proportion'])

        return {
            "optimizer": get_optimizer(
                [self.student_model],
                num_training_steps,
                self.yaml_cfg['optimizer']
            )
        }

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=1,
                          shuffle=True,
                          collate_fn=self.train_data.collate_fn,
                          num_workers=self.num_gpus*4)

    def val_dataloader(self):
        return DataLoader(self.eval_data,
                          batch_size=1,
                          collate_fn=self.eval_data.collate_fn,
                          num_workers=self.num_gpus*4)
    
    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=1,
                          collate_fn=self.test_data.collate_fn,
                          num_workers=self.num_gpus*4)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '-cfg', '--config', 
                        help='yaml config path for training')

    parser.add_argument('-t', '--test',
                        action='store_true', help='Enable testing mode')

    args = parser.parse_args()

    YAML_PATH = args.config or './data/distiller/ex.yaml'
    with open(YAML_PATH) as f:
        YAML_CFG = yaml.load(f, Loader = yaml.FullLoader)

    batch_size = YAML_CFG['train']['batch_size']
    output_dir = './results/pretrain/' + YAML_CFG['train']['output_dir']
    checkpoint = YAML_CFG['train']['checkpoint']
    gpus = YAML_CFG['train']['gpus']
    num_epochs = YAML_CFG['train']['num_epochs']
    use_fp16 = 16 if YAML_CFG['train']['use_fp16'] else 32
    accumulate_grad_batches = YAML_CFG['train']['accumulate_grad_batches']

    model = W2V2Distil(cfg = YAML_CFG)

    if checkpoint:
        model = model.load_from_checkpoint(output_dir + checkpoint)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
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
        gpus=gpus,
        strategy="ddp",
        # amp_backend="apex",
        precision=use_fp16,
        max_epochs=num_epochs,
        sync_batchnorm=True,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[early_stopping, checkpoint_callback],
    )

    if args.test:
        trainer.test(model)
    else:
        trainer.fit(model)

