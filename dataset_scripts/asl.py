import scipy.io
import os
from tqdm import tqdm
import sparse
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed



# Source data folder
path_dataset = '../datasets/ICCV2019_DVS_dataset/'
# Target data folder
path_dataset_dst = '../datasets/ICCV2019_DVS_dataset/clean_dataset_frames_2000/'


chunk_len_ms = 2
chunk_len_us = chunk_len_ms*1000
width = 240; height = 180


total_samples = [ s for d in os.listdir(path_dataset) for s in os.listdir(os.path.join(path_dataset, d)) ]
total_labels = [ s.split('_')[0] for s in total_samples ]

train_samples, test_samples = train_test_split(total_samples, test_size=0.2, random_state=0, stratify=total_labels)

        
    
def process_file_sample(path_dataset, label, f, train_samples):
    mode = 'train' if f in train_samples else 'test'
    filename_dst = path_dataset_dst + '/{}/'.format(mode) + '{}.pckl'.format(f[:-4])
    if os.path.isfile(filename_dst): return
    
    mat = scipy.io.loadmat(os.path.join(path_dataset,label, f))
    
    if mat['ts'][-1] < mat['ts'][0] > 200*1000: print(mat['ts'][-1], f)
    
    total_events = np.array([mat['x'], mat['y'], mat['ts'], mat['pol']]).transpose()[0]

    total_chunks = []
    while total_events.shape[0] > 0:
        end_t = total_events[-1][2]
        chunk_inds = np.where(total_events[:,2] >= end_t - chunk_len_us)[0]
        if len(chunk_inds) <= 4: 
            pass
        else:
            total_chunks.append(total_events[chunk_inds])
        total_events = total_events[:chunk_inds.min()]
    if len(total_chunks) == 0: 
        print('aaa')
        return
    total_chunks = total_chunks[::-1]
        
    total_frames = []
    for chunk in total_chunks:
        frame = sparse.COO(chunk[:,[1,0,3]].transpose().astype('int32'), 
                           np.ones(chunk.shape[0]).astype('int32'), 
                           (height, width, 2))   # .to_dense()
        total_frames.append(frame)
    total_frames = sparse.stack(total_frames)
    
    total_frames = np.clip(total_frames, a_min=0, a_max=255)
    total_frames = total_frames.astype('uint8')    
    
    if len(total_frames) > 200*1000 / chunk_len_us: print(mat['ts'][-1], f)
    
    pickle.dump(total_frames, open(filename_dst, 'wb'))


label = os.listdir(path_dataset)[0]
f = os.listdir(os.path.join(path_dataset, label))[0]

Parallel(n_jobs=8)(delayed(process_file_sample)(path_dataset, label, f, train_samples) for label in tqdm(os.listdir(path_dataset)) for f in tqdm(os.listdir(os.path.join(path_dataset, label))))


# %%

import os

path_dataset_dst = '../../datasets/ICCV2019_DVS_dataset/clean_dataset_frames_2000/'
class_mapping = { l:i for i,l in enumerate('a b c d e f g h i k l m n o p q r s t u v w x y'.split()) }

for f in os.listdir(path_dataset_dst + 'train'): os.rename(path_dataset_dst + 'train/' + f, path_dataset_dst + 'train/' + f.replace('.pckl', f'_{class_mapping[f.split("_")[0]]}.pckl'))
for f in os.listdir(path_dataset_dst + 'test'): os.rename(path_dataset_dst + 'test/' + f, path_dataset_dst + 'test/' + f.replace('.pckl', f'_{class_mapping[f.split("_")[0]]}.pckl'))


