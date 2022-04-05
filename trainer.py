import os

from torch import nn
import torch.nn.functional as F
import torch

import pytorch_lightning as pl

from pytorch_lightning import Trainer, LightningModule
from torch.optim import lr_scheduler

from data_generation import Event_DataModule
import evaluation_utils

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import training_utils
import json
import pandas as pd
import numpy as np
import copy
from torch.optim import AdamW

from models.EvT import CLFBlock, MLPBlock
from models.EvT import EvNetBackbone



class EvNetModel(LightningModule):

    def __init__(self, backbone_params, clf_params, optim_params, loss_weights=None):
        super().__init__()
        self.save_hyperparameters()

        self.backbone_params = backbone_params
        self.clf_params = clf_params
        self.optim_params = optim_params
        
        # Initialize Backbone
        self.backbone = EvNetBackbone(**backbone_params)
        # Initialize classifier
        self.clf_params['ipt_dim'] = self.backbone_params['embed_dim']
        # TODO: move to single variable
        self.models_clf = nn.ModuleDict([ [str(0),CLFBlock(**self.clf_params)] ])
        # self.models_clf = CLFBlock(**self.clf_params)
        
        self.loss_weights = loss_weights
        self.init_optimizers()
        
       
    def init_optimizers(self):
        self.criterion = nn.NLLLoss(weight = self.loss_weights)
        self.accuracy = pl.metrics.Accuracy()
        

    def forward(self, x, pixels):
        # Get updated latent vectors
        embs = self.backbone(x, pixels)
        # Get latent vectors classification
        clf_logits = torch.stack([ self.models_clf[str(0)](embs) ]).mean(axis=0)
        return embs, clf_logits
    
        
    def configure_optimizers(self):

        # Import base optimizer
        base_optim = AdamW
        optim = base_optim(self.parameters(), **self.optim_params['optim_params'])
    
        if 'scheduler' in self.optim_params: 
            if self.optim_params['scheduler']['name'] == 'lr_on_plateau': 
                sched = lr_scheduler.ReduceLROnPlateau(optim, **self.optim_params['scheduler']['params'])
            elif self.optim_params['scheduler']['name'] == 'one_cycle_lr': 
                sched = lr_scheduler.OneCycleLR(optim, max_lr=self.optim_params['optim_params']['lr'],  **self.optim_params['scheduler']['params'])
            return {'optimizer': optim, 'lr_scheduler': sched, 'monitor': self.optim_params['monitor']}
        return optim
    

    # Forward data and calculate loss and acc
    def step(self, polarity, pixels, y):
        embs, clf_logits = self(polarity, pixels)

        loss_clf, loss_contr = 0.0, 0.0
        logs = {}
            
        loss_clf = self.criterion(clf_logits, y)
        preds = torch.argmax(clf_logits, dim=-1)
        
        acc = self.accuracy(preds, y)
 
        logs['loss_clf'] = loss_clf
        logs['acc'] = acc
        
     
        logs['loss_total'] = loss_clf + loss_contr
        
        return logs
        

    def training_step(self, batch, batch_idx):
        #  batch_data -> (#imesteps, batch_size, #events, 2) - (#imesteps, batch_size, #events, 2) - (batch_size)
        polarity, pixels, y = batch    
        losses = self.step(polarity, pixels, y)
        for k,v in losses.items():
            self.log(f'train_{k}', v, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return losses['loss_total']


    def validation_step(self, batch, batch_idx):
        polarity, pixels, y = batch
        losses = self.step(polarity, pixels, y)
        for k,v in losses.items():
            self.log(f'val_{k}', v, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        
        return losses['loss_total']




def train(folder_name, path_results, data_params, backbone_params, clf_params, 
          training_params, optim_params, callback_params, logger_params):

    # Create the folder where to store the training results
    path_model = training_utils.create_model_folder(path_results, folder_name)
    
    callbacks = []
    for k, params in callback_params:
        if k == 'early_stopping': callbacks.append(EarlyStopping(**params))
        if k == 'lr_monitor': callbacks.append(LearningRateMonitor(**params))
        if k == 'model_chck': 
            params['dirpath'] = params['dirpath'].format(path_model)
            callbacks.append(ModelCheckpoint(**params))
        
    loggers = []
    if 'csv' in logger_params: 
        logger_params['csv']['save_dir'] = logger_params['csv']['save_dir'].format(path_model)
        loggers.append(CSVLogger(**logger_params['csv']))

    
    # =============================================================================
    # Train
    # =============================================================================
    dm = Event_DataModule(**data_params)
    backbone_params['token_dim'] = dm.token_dim
    clf_params['opt_classes'] = dm.num_classes
    
    if 'pos_encoding' in backbone_params and backbone_params['pos_encoding']['params'].get('shape', -1) == -1:
        backbone_params['pos_encoding']['params']['shape'] = (dm.width, dm.height)
    if backbone_params['downsample_pos_enc'] == -1: backbone_params['downsample_pos_enc'] = data_params['patch_size']
    
    if optim_params['scheduler']['name'] == 'one_cycle_lr':
        optim_params['scheduler']['params']['steps_per_epoch'] = 1
    
    model = EvNetModel(backbone_params=copy.deepcopy(backbone_params), 
                       clf_params=copy.deepcopy(clf_params), 
                       optim_params=copy.deepcopy(optim_params),
                       loss_weights = None if not data_params['balance'] else dm.train_dataloader().dataset.get_class_weights()
                       )
    
    trainer = Trainer(**training_params, callbacks=callbacks, logger=loggers)
    
    # Save all params
    json.dump({'data_params': data_params, 'backbone_params': backbone_params, 'clf_params': clf_params, 
               'training_params': training_params,
               'optim_params': optim_params, 'callbacks_params': callback_params, 'logger_params': logger_params},
              open(path_model+'all_params.json', 'w'))
    
    
    trainer.fit(model, dm)
    
    print(' ** Train finished:', path_model)

    logs = evaluation_utils.load_csv_logs_as_df(path_model)
    val_acc = logs[~logs['val_acc'].isna()]['val_acc']
    print(' - Max. Accuracy: {:.4f}'.format(val_acc.values.max()))
    
    for c in [ c for c in logs.columns if 'val_' in c and 'acc' not in c ]:
        v = logs[~logs[c].isna()][c]
        v = v.values.min() if len(v) > 0 else 0.0
        print(' - Min. [{}]: {:.4f}'.format(c, v))
    print("path_model = '{}'".format(path_model))
    
    return path_model

    
