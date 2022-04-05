import torch
from data_generation import Event_DataModule

from pytorch_lightning.metrics import Accuracy
import pandas as pd
import json
from tqdm import tqdm
import time
import numpy as np

import evaluation_utils
from trainer import EvNetModel


device = 'cuda:0'
# device = 'cpu'


path_model = './pretrained_models/DVS128_10_24ms_dwn/'
# path_model = './pretrained_models/DVS128_11_24ms_dwn/'
# path_model = './pretrained_models/SLAnimals_4s_48ms_dwn/'
# path_model = './pretrained_models/SLAnimals_3s_48ms_dwn/'
# path_model = './pretrained_models/ASL_DVS_dwn/'


path_weights = evaluation_utils.get_best_weigths(path_model, 'val_acc', 'max')
evaluation_utils.plot_training_evolution(path_model)
all_params = json.load(open(path_model + '/all_params.json', 'r'))
model = EvNetModel.load_from_checkpoint(path_weights, map_location=torch.device('cpu'), **all_params).eval().to(device)

def get_params(model):
    total_params = pd.DataFrame([ (n.split('.')[0],p.numel()/1000000) for n,p in model.backbone.named_parameters() if p.requires_grad ]).groupby(0).sum().sum().iloc[0]
    pos_encoding_params = pd.DataFrame([ (n.split('.')[0],p.numel()/1000000) for n,p in model.backbone.named_parameters() if p.requires_grad ]).groupby(0).sum().loc['pos_encoding'].iloc[0]
    stats = {
        'total_params': total_params,
        'backbone_params': total_params - pos_encoding_params,
        'pos_encoding_params': pos_encoding_params
        }
    return stats

print('\n\n ** Calculating parameter statistics')
param_stats = get_params(model)

# %%

def get_complexity_stats(model, all_params):
    data_params = all_params['data_params']
    data_params['batch_size'] = 1
    data_params['pin_memory'] = False
    data_params['sample_repetitions'] = 1
    dm = Event_DataModule(**data_params)
    dl = dm.val_dataloader()
    
    # https://github.com/sovrasov/flops-counter.pytorch
    from ptflops import get_model_complexity_info
    
    total_flops, total_macs, total_params, total_act_patches = [], [], [], []
    total_time_flops = []
    for polarity, pixels, labels in tqdm(dl):
        if polarity is None: continue
        polarity, pixels, labels = polarity.to(device), pixels.to(device), labels.to(device)
        
        for ts in range(len(polarity)):
            num_patches = sum(polarity[ts:ts+1].sum(-1).sum(0).sum(0) != 0)
            mask = polarity[ts:ts+1].sum(-1).sum(0).sum(0) != 0
            pol_t, pix_t = polarity[ts:ts+1][:,:,mask,:], pixels[ts:ts+1][:,:,mask,:]
            t = time.time()
            macs, params = get_model_complexity_info(model.backbone, 
                                               ({'kv': pol_t, 'pixels': pix_t},),
                                             input_constructor=lambda x: x[0],
                                             as_strings=False,
                                             print_per_layer_stat=False, verbose=False)
            total_time_flops.append(time.time() - t)
            flops = 2*macs
            total_act_patches.append(num_patches.cpu())
        total_flops.append(flops); total_macs.append(macs); total_params.append(params)
        
    return np.mean(total_flops), np.mean(total_act_patches)


print('\n\n ** Calculating complexity statistics')
flops, activated_patches = get_complexity_stats(model, all_params)


# %%

# =============================================================================
# Time analysis
# =============================================================================

def get_time_accuracy_stats(model, all_params):
    data_params = all_params['data_params']
    data_params['batch_size'] = 1
    data_params['pin_memory'] = False
    data_params['sample_repetitions'] = 1
    dm = Event_DataModule(**data_params)
    dl = dm.val_dataloader()
        
    total_time = []
    y_true, y_pred = [], []
    for polarity, pixels, labels in tqdm(dl):
        if polarity is None: continue
        polarity, pixels, labels = polarity.to(device), pixels.to(device), labels.to(device)
        t = time.time()
        embs, clf_logits = model(polarity, pixels)
        total_time.append((time.time() - t)/len(polarity))
        
        y_true.append(labels[0])
        y_pred.append(clf_logits.argmax())
    y_true, y_pred = torch.stack(y_true).to("cpu"), torch.stack(y_pred).to("cpu")
    acc_score = Accuracy()(y_true, y_pred).item()
    
    logs = evaluation_utils.load_csv_logs_as_df(path_model)
    train_acc = logs[~logs['val_acc'].isna()]['val_acc'].max()

    return np.mean(total_time)*1000, train_acc, acc_score

print('\n\n ** Calculating time and accuracy statistics')
avg_time, train_acc, val_acc = get_time_accuracy_stats(model, all_params)


# %%

print(f' - Model parameters: {param_stats["total_params"]:.2f} M | pos_encoding_parameters: {param_stats["pos_encoding_params"]:.2f} M | backbone_parameters: {param_stats["backbone_params"]:.2f} M ')
print(f' - Model FLOPs: {flops*1e-9:.2f} G')
print(f' - Average activated patches in [{all_params["data_params"]["dataset_name"]}]: {activated_patches:.1f}')
print(f' - Average processing time per time-window in device [{device}]: {avg_time:.4f} ms')
print(f' - Validation accuracy reported during training: {train_acc*100:.2f} %')
print(f' - Validation accuracy reported after training: {val_acc*100:.2f} %')


