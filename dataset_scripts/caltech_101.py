import os
import numpy as np
import sparse
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm


width, height = 240, 180
chunk_len_ms = 2
chunk_len_us = chunk_len_ms*1000
size = 0.25


# Source data folder
path_dataset = '../datasets/N_Caltech_101/Caltech101/'
# Target data folder
path_dataset_dst = '../datasets/N_Caltech_101/clean_dataset_frames_2000/'

if not os.path.isfile(path_dataset_dst):
    os.mkdir(path_dataset_dst)
    os.mkdir(path_dataset_dst + '/train')
    os.mkdir(path_dataset_dst + '/test')

sample_files = [ os.path.join(path_dataset, cat, fs) for cat in os.listdir(path_dataset) for fs in os.listdir(os.path.join(path_dataset, cat)) ]

train_samples, test_samples = train_test_split(sample_files, test_size=size, random_state=0,
                                      stratify=[ f.split('/')[-2] for f in sample_files ])

max_values = []
class_folders = os.listdir(path_dataset)
for cat_id, cat in enumerate(tqdm(class_folders)):
    
    file_samples = os.listdir(os.path.join(path_dataset, cat))
    for fs in file_samples:
            
        with open(os.path.join(path_dataset, cat, fs), "rb") as f:
           
            mode = 'train' if os.path.join(path_dataset, cat, fs) in train_samples else 'test'
            filename_dst = path_dataset_dst + f'{mode}/' + f'{cat}_{fs.replace(".bin", "")}_{cat_id}.pckl'
            
            raw_data = np.fromfile(open(os.path.join(path_dataset, cat, fs), 'rb'), dtype=np.uint8)
            
            raw_data = np.uint32(raw_data)
            all_y = raw_data[1::5]
            all_x = raw_data[0::5]
            all_p = (raw_data[2::5] & 128) >> 7 #bit 7
            all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])


            total_events = np.array([all_x, all_y, all_ts, all_p]).transpose()
            max_values.append(total_events.max(0))

            total_chunks = []
            while total_events.shape[0] > 0:
                end_t = total_events[-1][2]
                chunk_inds = np.where(total_events[:,2] >= end_t - chunk_len_us)[0]
                if len(chunk_inds) <= 4: 
                    pass
                else:
                    total_chunks.append(total_events[chunk_inds])
                total_events = total_events[:chunk_inds.min()-1]
            if len(total_chunks) == 0: continue
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
        
            pickle.dump(total_frames, open(filename_dst, 'wb'))
     
