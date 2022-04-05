import aermanager
from aermanager.aerparser import load_events_from_file
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os


# Source data folder
path_dataset_src = '../datasets/DvsGesture/'
# Target data folder
path_dataset_dst = '../datasets/DvsGesture/clean_dataset/'


train_files, test_files = 'trials_to_train.txt', 'trials_to_test.txt'
with open(path_dataset_src + train_files, 'r') as f: train_files = f.read().splitlines()
with open(path_dataset_src + test_files, 'r') as f: test_files = f.read().splitlines()


# Load and stor dataset event samples
def store_samples(events_files, mode):
    for events_file in tqdm(events_files):
        if events_file == '': continue
        labels = pd.read_csv(path_dataset_src + events_file.replace('.aedat', '_labels.csv'))
        shape, events = load_events_from_file(path_dataset_src + events_file, parser=aermanager.parsers.parse_dvs_ibm)
        
        # Load user samples
        # Class 0 for non action
        time_segment_class = []     # [(t_init, t_end, class)]
        prev_event = 0
        for _,row in labels.iterrows():
            if (row.startTime_usec-1 - prev_event) > 0: time_segment_class.append((prev_event, row.startTime_usec-1, 0, (row.startTime_usec-1 - prev_event)/1000))
            if ((row.endTime_usec - row.startTime_usec)) > 0: time_segment_class.append((row.startTime_usec, row.endTime_usec, row['class'], (row.endTime_usec - row.startTime_usec)/1000))
            prev_event = row.endTime_usec + 1 
        time_segment_class.append((prev_event, np.inf, 0))
        
        total_events = []
        curr_event = []
        # for e in tqdm(events):
        for i, e in enumerate(events):
            if e[2] >= time_segment_class[0][1]:
                # Store event
                if len(curr_event) > 1: total_events.append((time_segment_class[0], len(curr_event), np.array(curr_event)))
                else: print(' ** EVENT ERROR:', events_file.replace('.aedat','') + '_num{:02d}_label{:02d}.pckl'.format(i, time_segment_class[0][2]), time_segment_class[0], len(curr_event))
                time_segment_class = time_segment_class[1:]
                curr_event = []
            curr_event.append(list(e))
            
        if len(curr_event) > 1: total_events.append((time_segment_class[0], len(curr_event), np.array(curr_event)))
        time_segment_class = time_segment_class[1:]
        curr_event = []
        
        for i, (meta, _, events) in enumerate(total_events):
            pickle.dump((events, meta[2]), 
                        open(os.path.join(path_dataset_dst, mode, events_file.replace('.aedat','') + '_num{:02d}_label{:02d}.pckl'.format(i, meta[2])), 'wb'))


store_samples(train_files, 'train')
store_samples(test_files, 'test')

