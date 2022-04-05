import os
os.chdir('..')
import pandas as pd
import numpy as np
import sparse
import pickle

import aermanager
from aermanager.aerparser import load_events_from_file
from sklearn.model_selection import train_test_split
from tqdm import tqdm


np.random.seed(0)

chunk_len_ms = 2
chunk_len_us = chunk_len_ms*1000
height, width = 128, 128
size = 0.25


# Source data folder
path_dataset = './datasets/SL_Animals/'
files = os.listdir(path_dataset + 'allusers_aedat/')
parser = aermanager.parsers.parse_dvs_128


# Target data folder
if not os.path.isdir(path_dataset + 'SL_animal_splits'):
    os.mkdir(path_dataset + 'SL_animal_splits')
    os.makedirs(path_dataset + 'SL_animal_splits/' + f'dataset_4sets_{chunk_len_us}/train')
    os.makedirs(path_dataset + 'SL_animal_splits/' + f'dataset_4sets_{chunk_len_us}/test')
    os.makedirs(path_dataset + 'SL_animal_splits/' + f'dataset_3sets_{chunk_len_us}/train')
    os.makedirs(path_dataset + 'SL_animal_splits/' + f'dataset_3sets_{chunk_len_us}/test')


train_samples_4sets, test_samples_4sets = train_test_split(files, test_size=size, random_state=0,
                                      stratify=[ f[:-6].split('_')[-1] for f in files ])

train_samples_3sets, test_samples_3sets = train_test_split([ f for f in files if '_indoor' not in f ], test_size=size, random_state=0,
                                      stratify=[ f[:-6].split('_')[-1] for f in files if '_indoor' not in f])


for events_file in tqdm(files):
    shape, events = load_events_from_file(path_dataset + 'allusers_aedat/' + events_file, parser=parser)
    labels = pd.read_csv(path_dataset + 'tags_updated_19_08_2020/' + events_file.replace('.aedat', '.csv'))
    filename_dst = path_dataset + 'SL_animal_splits/' + f'dataset_{{}}_{chunk_len_us}/{{}}/' + \
        events_file.replace('.aedat', '_{}_{}.pckl')

    for _,row in labels.iterrows():
        
        sample_events = events[row.startTime_ev:row.endTime_ev]
        
        total_events = np.array([sample_events['x'], sample_events['y'], sample_events['t'], sample_events['p']]).transpose()

        total_chunks = []
        while total_events.shape[0] > 0:
            end_t = total_events[-1][2]
            chunk_inds = np.where(total_events[:,2] >= end_t - chunk_len_us)[0]
            if len(chunk_inds) <= 4: 
                pass
            else:
                total_chunks.append(total_events[chunk_inds])
            total_events = total_events[:max(1, chunk_inds.min())-1]
        if len(total_chunks) == 0: continue
        total_chunks = total_chunks[::-1]
            
            
        total_frames = []
        for chunk in total_chunks:
            frame = sparse.COO(chunk[:,[0,1,3]].transpose().astype('int32'), 
                               np.ones(chunk.shape[0]).astype('int32'), 
                               (height, width, 2))   # .to_dense()
            total_frames.append(frame)
        total_frames = sparse.stack(total_frames)
        
        total_frames = np.clip(total_frames, a_min=0, a_max=255)
        total_frames = total_frames.astype('uint8')

        if '_sunlight' in events_file:  val_set = 'S4' # S4 indoors with frontal sunlight
        elif '_indoor' in events_file:  val_set = 'S3' # S3 indoors neon light
        elif '_dc' in events_file:      val_set = 'S2' # S2 natural side light
        elif '_imse' in events_file:    val_set = 'S1' # S1 natural side light
        else: raise ValueError('Set not handled')

        if events_file in train_samples_3sets:
            pickle.dump(total_frames, open(filename_dst.format('3sets', 'train', val_set, row['class']), 'wb'))
        if events_file in test_samples_3sets:
            pickle.dump(total_frames, open(filename_dst.format('3sets', 'test', val_set, row['class']), 'wb'))
        if events_file in train_samples_4sets:
            pickle.dump(total_frames, open(filename_dst.format('4sets', 'train', val_set, row['class']), 'wb'))
        if events_file in test_samples_4sets:
            pickle.dump(total_frames, open(filename_dst.format('4sets', 'test', val_set, row['class']), 'wb'))
        
        
        
