import numpy as np
from tqdm import tqdm
import pickle
import os
import sparse
os.chdir('..')



chunk_len_ms = 12
chunk_len_us = chunk_len_ms*1000
height = width = 128
mode = 'train'

# Read dataset filenames
if mode == 'train':
    # Source data folder
    path_dataset_src = './datasets/DvsGesture/clean_dataset/train/'
    # Target data folder
    path_dataset_dst = './datasets/DvsGesture/clean_dataset_frames_{}/train/'.format(chunk_len_us)
else:
    path_dataset_src = './datasets/DvsGesture/clean_dataset/test/'
    path_dataset_dst = './datasets/DvsGesture/clean_dataset_frames_{}/test/'.format(chunk_len_us)

event_files = os.listdir(path_dataset_src)
if not os.path.isdir(path_dataset_dst): os.makedirs(path_dataset_dst)


# %%


for ef in tqdm(event_files):
    
    total_events, label = pickle.load(open(path_dataset_src+ef, 'rb'))
    total_events = total_events.astype('int32')
    
    total_chunks = []
    while total_events.shape[0] > 0:
        end_t = total_events[-1][2]
        chunk_inds = np.where(total_events[:,2] >= end_t - chunk_len_us)[0]
        if len(chunk_inds) <= 4: 
            pass
        else:
            total_chunks.append(total_events[chunk_inds])
        total_events = total_events[:max(1, chunk_inds.min())-1]
    if len(total_chunks) == 0: 
        print('aaa')
        continue
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

    pickle.dump(total_frames, open(path_dataset_dst + ef, 'wb'))
    



