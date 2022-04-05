import torch
from data_generation import Event_DataModule

from pytorch_lightning.metrics import Accuracy
from sklearn.metrics import confusion_matrix
import pandas as pd
import json
from tqdm import tqdm
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os


def get_best_weigths(path_model, metric, mode):
    assert mode in ['min', 'max']
    mode = max if mode == 'max' else min
    w = os.listdir(os.path.join(path_model, 'weights'))
    path_weights = mode(w, key=lambda x: [ float(s[len(metric)+1:len(metric)+1+7]) for s in x.split('-') if s.startswith(metric) ][0])
    return os.path.join(path_model, 'weights',path_weights)


def load_csv_logs_as_df(path_model):
    log_file = path_model + '/train_log/version_0/metrics.csv'
    logs = pd.read_csv(log_file)
    for i, row in logs[logs.epoch.isna()].iterrows():
        candidates = logs[(~logs.epoch.isna()) & (logs.step >= int(row.step))].epoch.min()
        logs.loc[i, 'epoch'] = candidates
    return logs
    
def plot_training_evolution(path_model):

    logs = load_csv_logs_as_df(path_model)
    
    lr_col = [ c for c in logs.columns if 'lr' in c ][0]
    lr = logs[~logs[lr_col].isna()][lr_col]
    c = [ c for c in logs.columns if 'val_acc' in c ][0]
    val_acc = logs[~logs[c].isna()][c]

    fig, ax1 = plt.subplots(figsize=(12,6), dpi=200)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    for c in [ c for c in logs.columns if 'val_' in c and 'acc' not in c ]:
        loss = logs[~logs[c].isna()][c]
        ax1.plot(loss.values, label=c)
        
    ax2.plot(val_acc.values, 'g')
    ax3.plot(lr.values, 'r')
    
    if 'ASL' in path_model: ax2.set_ylim(0.95, 1)  # Acc lims
    else: ax2.set_ylim(0.5, 1)  # Acc lims

    ax1.set_ylabel('val_loss', color='b', fontsize=18)
    ax2.set_ylabel('val_acc', color='g', fontsize=18)
    ax3.set_ylabel('lr', color='r', fontsize=18)

    ax2.hlines(val_acc.max(), 0, len(val_acc.values), color='g', linestyle='--', alpha=0.7)
    ax1.hlines(logs[~logs['val_loss_total'].isna()]['val_loss_total'].min(), 0, len(val_acc.values), color='y', linestyle='--', alpha=0.7)

    
    ax3.spines['right'].set_position(('outward', 60))
    fig.tight_layout()

    plt.title('{} | Acc.: {:.4f} | Loss: {:.4f}'.format(' | '.join(path_model.split('/')[-3:]), val_acc.max(), 
              logs[~logs['val_loss_total'].isna()]['val_loss_total'].min()), fontsize=16)
    ax1.legend()
    
    plt.show()


def get_evaluation_results(path_model, path_weights, skip_validation=False, force=False, device='cpu'):
    
    all_params = json.load(open(path_model + '/all_params.json', 'r'))
    stats_filename = path_model + '/stats_validation.json'
    cm_filename = path_model + '/confussion_matrix.pckl'
    if not force and os.path.isfile(stats_filename):
        df_cm = pickle.load(open(cm_filename, 'rb')) if os.path.isfile(cm_filename) else None
        return all_params, json.load(open(stats_filename, 'r')),df_cm
    
    stats = {}
    logs = load_csv_logs_as_df(path_model)
    c = [ c for c in logs.columns if 'val_acc' in c ][0]
    stats['training_val_acc'] = logs[~logs[c].isna()][c].max()
    stats['training_val_loss'] = logs[~logs['val_loss_total'].isna()]['val_loss_total'].min()
    stats['training_val_loss_total'] = logs[~logs['val_loss_total'].isna()]['val_loss_total'].min()
    stats['num_epochs'] = int(logs['epoch'].max())
    data_params = all_params['data_params']
    stats['chunk_len_ms'] = data_params['chunk_len_ms']
        
    # Load model and logs
    if not skip_validation:
        from trainer import EvNetModel
        # model = EvNetModel.load_from_checkpoint(path_weights, map_location=torch.device('cpu')).eval().to(device)
        model = EvNetModel.load_from_checkpoint(path_weights, map_location=torch.device('cpu'), **all_params).eval().to(device)
    
        # Load data
        data_params['batch_size'] = 1
        data_params['pin_memory'] = False
        # data_params['workers'] = 1
        dm = Event_DataModule(**data_params)
        dl = dm.val_dataloader()
        print();print(' * Loading val data')
        val_data = [ d for d in tqdm(dl) ]
        
        # Get predictions and accuracy
        print();print(' * Evaluating...')
        t = time.time()
        y_true, y_pred = [], []
        for polarity, pixels, labels in tqdm(val_data):
            polarity, pixels, label = polarity.to(device), pixels.to(device), labels.to(device)
            embs, clf_logits = model(polarity, pixels)
            y_true.append(label[0])
            y_pred.append(clf_logits.argmax())
        t = time.time() - t
        y_true, y_pred = torch.stack(y_true).to("cpu"), torch.stack(y_pred).to("cpu")
        acc_score = Accuracy()(y_true, y_pred).item()
    
        # Get stats
        total_time_ms = t*1000
        num_samples = len(val_data)
        total_chunks = [ d[0].shape[0] for d in val_data ]
        num_chunks = sum(total_chunks)
        total_events = [ d[0].shape[2] for d in val_data ]
    
        stats['validation_val_acc'] = acc_score
        stats['sequence_ms'] = total_time_ms/num_samples
        stats['chunk_ms'] = total_time_ms/num_chunks
        stats['events_per_chunk'] = {'mean': np.mean(total_events), 'median': np.median(total_events), 'p05': np.percentile(total_events, 5), 'p95': np.percentile(total_events, 95)}
        stats['ms/ms'] = stats['chunk_ms'] / data_params['chunk_len_ms']
    
        print('*'*40)
        print('*'*40)
        print(stats)
        print('*'*40)
        print('*'*40)
    
    
        # Calculate confussion matrix
        class_mapping = dm.class_mapping
        class_mapping = { k:'{}. {}'.format(k,v) for k,v in class_mapping.items() }
        labels_mapping = { v:k for k,v in dl.dataset.unique_labels.items() }
        y_true_cm = [ class_mapping[l] for l in y_true.numpy() ]
        y_pred_cm = [ class_mapping[l] for l in y_pred.numpy() ]
        labels = sorted(set(y_true_cm), key=lambda x: int(x.split()[0][:-1]))
        cm = confusion_matrix(y_true_cm, y_pred_cm, normalize='true', labels=labels)
        df_cm = pd.DataFrame(cm, index = labels, columns = labels)
        pickle.dump(df_cm, open(cm_filename, 'wb'))
        print('*'*40)
        print('*'*40)
        print(df_cm)
        print('*'*40)
        print('*'*40)        
    
    else:
        def_val = 0.0
        stats['validation_val_acc'] = def_val
        stats['sequence_ms'] = def_val
        stats['chunk_ms'] = def_val
        stats['events_per_chunk'] = def_val
        stats['ms/ms'] = def_val
        df_cm = None

        
    return all_params, stats, df_cm




