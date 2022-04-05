#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:26:41 2021

@author: asabater
"""

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch
# from torch.nn.utils.rnn import pad_sequence

import os
import pickle
import numpy as np
import json
from skimage.util import view_as_blocks
import copy
from scipy import ndimage


class_mapping = {0: 'background', 1: 'hand_clapping', 2: 'right_hand_wave', 
                 3: 'left_hand_wave', 4: 'right_arm_clockwise', 
                 5: 'right_arm_counter_clockwise', 6: 'left_arm_clockwise', 
                 7: 'left_arm_counter_clockwise', 8: 'arm_roll', 
                 9: 'air_drums', 10: 'air_guitar', 11: 'other_gestures'}


# - TODO: split event sequences by chunks
# - TODO: zero-pad chunks
# - TODO: zero-pad batches
# TODO: ensure events sorted by time

# - TODO: remove last chunk (not complete)
# - TODO: create chunks by filtering the array, not iterating over events. -> move array to torch tensor

# TODO: BatchSampler -> generates indices per batch. Create custom para samplear por class y duplicados (anchoring)
# TODO: change all the np.random to torch.random

# class DVS128Dataset(Dataset):
#     def __init__(self, samples_folder, chunk_len_ms, height=128, width=128, 
#                  skip_last_event=False, classes_to_exclude=[], transform=None):

#         self.samples_folder = samples_folder
#         self.chunk_len_ms = chunk_len_ms
#         self.chunk_len_us = chunk_len_ms*1000
#         self.height = height
#         self.width = width
#         self.skip_last_event = skip_last_event
#         self.transform = transform

#         self.samples = os.listdir(samples_folder)
#         for l in classes_to_exclude:
#             self.samples = [ s for s in self.samples if '_label{:02}'.format(l) not in s ]
        
#         self.labels = np.array([ int(t[5:7]) for s in self.samples for t in s.split('_') if 'label' in t ]).astype('int8')
#         unique_labels = { l:i for i,l in enumerate(set(self.labels)) }
#         self.labels = [ unique_labels[l] for l in self.labels ]
#         self.num_classes = len(unique_labels)

#     def __len__(self):
#         return len(self.samples)
    
#     def get_label_dict(self):
#         label_dict = { c:[] for c in set(self.labels) }
#         for i,l in enumerate(self.labels): label_dict[l].append(i)
#         for k in label_dict: label_dict[k] = torch.IntTensor(label_dict[k])
#         return label_dict
    
#     # event_chunk -> [N, 4], [x, y, t, p] x N -> (p0,p1) x N, (x,y) x N
#     def aggregate_events_array_per_pixel(self, event_chunk):
#         # [N, 2] -> (pixel_num, p) x N
#         polarity = torch.nn.functional.one_hot(event_chunk[:,3].long(), num_classes=2)
#         # (N)
#         pixels = event_chunk[:,0] + event_chunk[:,1]*self.width    
    
#         # Aggreagate by pixels and count polarities
#         unique_pixels, pixel_inds = pixels.unique(dim=0, return_inverse=True)
#         pixel_inds = pixel_inds.view(pixels.size(0), 1).expand(-1, polarity.size(1))
#         unique_pixels = unique_pixels.view(unique_pixels.size(0), 1).expand(-1, polarity.size(1))
#         agg_polarity_pixel_count = torch.zeros_like(unique_pixels, dtype=torch.long).scatter_add_(0, pixel_inds, polarity)
        
#         # Transform unique pixels to x/y
#         unique_pixels = torch.column_stack([torch.remainder(unique_pixels[:,0], self.width), unique_pixels[:,0] // self.width])
        
#         return agg_polarity_pixel_count, unique_pixels
        
#     # Split events into chunks
#     def get_total_chunks_by_iteration(self, total_events):
#         total_chunks = []; init_t = total_events[0][2]; curr_chunk = []
#         for e in total_events:
#             if init_t + self.chunk_len_us < e[2]:
#                 total_chunks.append(torch.stack(curr_chunk, axis=0))
#                 curr_chunk, init_t = [], e[2]
#             curr_chunk.append(e)
#         if not self.skip_last_event:
#             total_chunks.append(torch.stack(curr_chunk, axis=0))
#             curr_chunk = []
#         return total_chunks
#     def get_total_chunks_by_filtering(self, total_events):
#             total_chunks = []
#             while total_events.shape[0] > 0:
#                 init_t = total_events[0][2]
#                 chunk_inds = torch.where(total_events[:,2] <= init_t + self.chunk_len_us)
#                 total_chunks.append(total_events[chunk_inds])
#                 total_events = total_events[chunk_inds[0].max()+1:]
#             return total_chunks
#     def extract_chunk_data(self, total_chunks):
#         chunk_data = [ self.aggregate_events_array_per_pixel(curr_chunk) for curr_chunk in total_chunks ]
#         return [ cd[0] for cd in chunk_data ], [ cd[1] for cd in chunk_data ]
            
#     def __getitem__(self, idx):
#         filename = self.samples[idx]
#         label = self.labels[idx]
        
#         total_events = pickle.load(open(os.path.join(self.samples_folder + filename), 'rb'))  # (events, label)
#         # total_events = total_events[0]      # (x,y,t,p)
#         total_events = torch.Tensor(total_events[0].astype('int32'))      # (x,y,t,p)

#         if self.transform:
#             # sample = self.transform(sample)
#             raise ValueError('Transformations not implemented')
            
#         else:
            
#             # total_chunks = self.get_total_chunks_by_iteration(total_events)  # 15.18s
#             total_chunks = self.get_total_chunks_by_filtering(total_events)  # 404.74ms
#             total_polarity, total_pixels = self.extract_chunk_data(total_chunks) # 4.93 ms

                    
#         return total_polarity, total_pixels, label


class DVS128Dataset_from_frames(Dataset):
    def __init__(self, samples_folder, chunk_len_ms, 
                 validation,
                 # max_sample_len_ms = -1,
                 augmentation_params,
                 preproc_polarity, patch_size, min_activations_per_patch,
                 bins,
                 min_patches_per_chunk, min_events_per_chunk, num_extra_chunks,
                 dataset_name, height, width, 
                 classes_to_exclude=[]):
        
        print(' * Creating DVS128Dataset_from_frames. Validation:', validation)

        self.samples_folder = samples_folder
        self.validation = validation
        self.chunk_len_ms = chunk_len_ms
        self.chunk_len_us = chunk_len_ms*1000
        
        self.sparse_frame_len_us = int(self.samples_folder.split('/')[-3].split('_')[-1])     # len of each loaded sparse frame
        self.sparse_frame_len_ms = self.sparse_frame_len_us // 1000
        assert  self.chunk_len_us % self.sparse_frame_len_us == 0
        self.chunk_size = self.chunk_len_us // self.sparse_frame_len_us             # Size of the grouped frame chunks
        
        self.height = height
        self.width = width
        self.min_patches_per_chunk = min_patches_per_chunk
        self.min_events_per_chunk = min_events_per_chunk
        self.num_extra_chunks = num_extra_chunks
        
        # Define data augmentation functions
        print(augmentation_params, 'max_sample_len_ms' in augmentation_params, augmentation_params['max_sample_len_ms'] != -1)
        # self.crop_in_time, self.crop_in_space, self.drop_token, self.random_shift, self.crop_events = None, None, None, None, None
        self.augmentation_params = augmentation_params
        if augmentation_params is not None and len(augmentation_params) != 0:
            if 'max_sample_len_ms' in augmentation_params and augmentation_params['max_sample_len_ms'] != -1:
                # assert augmentation_params['max_sample_len_ms'] % self.sparse_frame_len_ms == 0
                self.num_sparse_frames = augmentation_params['max_sample_len_ms'] // self.sparse_frame_len_ms
                # self.crop_in_time = self.get_crop_in_time_func(augmentation_params['max_sample_len_ms'])
            if 'random_frame_size' in augmentation_params and augmentation_params['random_frame_size'] is not None:
                self.x_lims = (int(width*augmentation_params['random_frame_size']), width)
                self.y_lims = (int(height*augmentation_params['random_frame_size']), height)
                # self.crop_in_space = self.get_crop_in_space_func((int(height*augmentation_params['random_frame_size']), height), 
                #                                             (int(width*augmentation_params['random_frame_size']), width))
            if 'drop_token' in augmentation_params and augmentation_params['drop_token'][0] != 0.0:
                self.drop_perc, self.drop_mode = augmentation_params['drop_token']
                # self.drop_token = self.get_drop_token_function(*augmentation_params['drop_token'])
            if 'random_shift' in augmentation_params and augmentation_params['random_shift']:
                # self.random_shift = self.get_shift_func()
                pass
            if 'crop_to_max_events' in augmentation_params and augmentation_params['crop_to_max_events'] is not None:
                raise ValueError('Not Implemented')
                min_crop = min([ min(s) for s in augmentation_params['random_frame_size'] ]) if self.crop_in_space else min(height, width)
                med_frame_size = 128-(128-min_crop)//2
                res = json.load(open('./datasets/DvsGesture/dataset_stats_{}.json'.format(med_frame_size), 'r'))
                max_events = int(res[str(self.chunk_len_ms)][augmentation_params['crop_to_max_events']])
                if self.drop_token: max_events = int(max_events*(1-augmentation_params['drop_token'][0]))     
                self.max_events_per_chunk = max_events
                # self.crop_events = self.get_crop_to_max_events(max_events)
            self.h_flip = augmentation_params.get('h_flip', False)
                
        # print('+'*20, self.crop_in_time, self.crop_in_space, self.drop_token, self.random_shift)
        
        self.bins = bins
        self.preproc_polarity = preproc_polarity
        self.patch_size = patch_size
        self.original_event_size = 1 if '1' in self.preproc_polarity else 2
        self.preproc_event_size = self.original_event_size*bins
        self.token_dim = patch_size*patch_size * self.preproc_event_size
        
        if min_activations_per_patch > 0 and min_activations_per_patch <= 1: 
            self.min_activations_per_patch = int(min_activations_per_patch*patch_size*patch_size+1)
        else: self.min_activations_per_patch = 0
        print(f' * patch_size {patch_size}x{patch_size} [{patch_size*patch_size}] | min_activations {self.min_activations_per_patch}')
        
        
        self.height = height
        self.width = width

        self.samples = os.listdir(samples_folder)
        if dataset_name == 'DVS128':
            for l in classes_to_exclude:
                self.samples = [ s for s in self.samples if '_label{:02}'.format(l) not in s ]
            self.labels = np.array([ int(t[5:7]) for s in self.samples for t in s.split('_') if 'label' in t ]).astype('int8')
            self.unique_labels = { l:i for i,l in enumerate(sorted(set(self.labels))) }
            self.labels = [ self.unique_labels[l] for l in self.labels ]
            self.num_classes = len(self.unique_labels)
        elif dataset_name in ['ASL_DVS', 'HMDB', 'UCF101', 'UCF50', 'SLAnimals_4s', 'SLAnimals_3s', 'N_Cars', 'Caltech']:
            self.labels = [ s.split('_')[-1][:-5] for s in self.samples ]
            self.unique_labels = { l:i for i,l in enumerate(sorted(set(self.labels))) }
            self.labels = [ self.unique_labels[l] for l in self.labels ]
            self.num_classes = len(self.unique_labels)
        else: raise ValueError(f'dataset_name [{dataset_name}] not handled')
            
        
    def get_class_weights(self):
        label_dict = self.get_label_dict()
        label_dict = { k:label_dict[k] for k in sorted(label_dict) }
        num_samples = sum([ len(v) for v in label_dict.values() ])
        # max_len = max([ len(v) for v in label_dict.values() ])
        # class_weigths = { k:max_len/len(v) for k,v in label_dict.items() }
        # class_weigths = { k:(len(v), num_samples/(len(label_dict)*len(v))) for k,v in label_dict.items() }
        # class_weigths = { k:num_samples/(len(label_dict)*len(v)) for k,v in label_dict.items() }
        class_weigths = [ num_samples/(len(label_dict)*len(v)) for k,v in label_dict.items() ]
        return torch.tensor(class_weigths)
             
        
    # Crop sequence to self.num_sparse_frames
    # def get_crop_in_time_func(self, max_sample_len_ms):
    #     assert max_sample_len_ms % self.sparse_frame_len_ms == 0
    #     num_sparse_frames = max_sample_len_ms // self.sparse_frame_len_ms
    #     # print('num_sparse_frames', num_sparse_frames)
    #     def crop(total_events):
    #         # print('Cropping:', len(total_events))
    #         if len(total_events) > num_sparse_frames:
    #             if not self.validation:     # Crop sequence randomly
    #                 init = np.random.randint(len(total_events) - num_sparse_frames)
    #                 end = init + num_sparse_frames
    #                 total_events = total_events[init:end]
    #             else:                       # Crop to the middle part
    #                 init = (len(total_events) - num_sparse_frames) // 2
    #                 end = init + num_sparse_frames
    #                 total_events = total_events[init:end]
    #         # assert len(total_events) < num_sparse_frames, str(len(total_events)) + '  ' + str(num_sparse_frames)
    #         return total_events
    #     # print('Return crop func')
    #     return crop
    def crop_in_time(self, total_events):
            # print('Cropping:', len(total_events))
            if len(total_events) > self.num_sparse_frames:
                if not self.validation:     # Crop sequence randomly
                    init = np.random.randint(len(total_events) - self.num_sparse_frames)
                    end = init + self.num_sparse_frames
                    total_events = total_events[init:end]
                else:                       # Crop to the middle part
                    init = (len(total_events) - self.num_sparse_frames) // 2
                    end = init + self.num_sparse_frames
                    total_events = total_events[init:end]
            # assert len(total_events) < num_sparse_frames, str(len(total_events)) + '  ' + str(num_sparse_frames)
            return total_events
        
    
    # Crop sequence in space
    # x_lims/y_lims -> (min/max length) -> (length, not coordinates)
    # def get_crop_in_space_func(self, x_lims, y_lims):
    #     # print('get_crop_in_space_func', x_lims, y_lims)
    #     def crop(total_events):
    #         _, y_size, x_size, _ = total_events.shape
    #         # print((y_lims, x_lims), (y_size, x_size), total_events.shape)
    #         if not self.validation:     # Crop sequence randomly
    #             new_x_size = np.random.randint(x_lims[0], x_lims[1]+1)
    #             new_y_size = np.random.randint(y_lims[0], y_lims[1]+1)
                
    #             if self.patch_size != 1:
    #                 new_x_size -= new_x_size % self.patch_size
    #                 new_y_size -= new_y_size % self.patch_size
                
    #             x_init = np.random.randint(x_size - new_x_size+1); x_end = x_init + new_x_size
    #             y_init = np.random.randint(y_size - new_y_size+1); y_end = y_init + new_y_size
    #             # total_events = total_events[:, x_init:x_end, y_init:y_end, :]
    #             total_events = total_events[:, y_init:y_end, x_init:x_end, :]
    #         else:                       # Crop to the middle part
    #             new_x_size = (x_lims[0] + x_lims[1])//2
    #             new_y_size = (y_lims[0] + y_lims[1])//2
                
    #             if self.patch_size != 1:
    #                 new_x_size -= new_x_size % self.patch_size
    #                 new_y_size -= new_y_size % self.patch_size
                    
    #             x_init = (x_size - new_x_size)//2; x_end = x_init + new_x_size
    #             y_init = (y_size - new_y_size)//2; y_end = y_init + new_y_size
    #             # total_events = total_events[:, x_init:x_end, y_init:y_end, :]
    #             total_events = total_events[:, y_init:y_end, x_init:x_end, :]
    #         # print('total_events.shape', total_events.shape, (new_y_size, new_x_size))
    #         assert total_events.shape[1] == new_y_size and total_events.shape[2] == new_x_size
    #         return total_events
    #     return crop
    def crop_in_space(self, total_events):
        # print(type(total_events), len(total_events), total_events.shape)
        _, y_size, x_size, _ = total_events.shape
        # print(self.y_lims, '|', self.x_lims, '|', total_events.shape)
        if not self.validation:     # Crop sequence randomly
            new_x_size = np.random.randint(self.x_lims[0], self.x_lims[1]+1)
            new_y_size = np.random.randint(self.y_lims[0], self.y_lims[1]+1)
            
            if self.patch_size != 1:
                new_x_size -= new_x_size % self.patch_size
                new_y_size -= new_y_size % self.patch_size
            
            x_init = np.random.randint(x_size - new_x_size+1); x_end = x_init + new_x_size
            y_init = np.random.randint(y_size - new_y_size+1); y_end = y_init + new_y_size
            # total_events = total_events[:, x_init:x_end, y_init:y_end, :]
            total_events = total_events[:, y_init:y_end, x_init:x_end, :]
        else:                       # Crop to the middle part
            new_x_size = (self.x_lims[0] + self.x_lims[1])//2
            new_y_size = (self.y_lims[0] + self.y_lims[1])//2
            
            if self.patch_size != 1:
                new_x_size -= new_x_size % self.patch_size
                new_y_size -= new_y_size % self.patch_size
                
            x_init = (x_size - new_x_size)//2; x_end = x_init + new_x_size
            y_init = (y_size - new_y_size)//2; y_end = y_init + new_y_size
            # total_events = total_events[:, x_init:x_end, y_init:y_end, :]
            total_events = total_events[:, y_init:y_end, x_init:x_end, :]
        # print('total_events.shape', total_events.shape, (new_y_size, new_x_size))
        # print(total_events.shape, '|', new_y_size, y_init, y_end, '|', new_x_size, x_init, x_end)
        assert total_events.shape[1] == new_y_size and total_events.shape[2] == new_x_size, print(total_events.shape, new_y_size, new_x_size)
        return total_events
        
        
    
    # Remove random events from sequence based on percentage
    # drop_mode == 'fixed' -> drop same pixels for all the sequence
    # drop_mode == 'rand' -> drop random events in each time-step
    # def get_drop_token_function(self, drop_perc, drop_mode):
    #     def drop_token(total_events):
    #         if self.validation: 
    #             return total_events
    #         if drop_mode == 'rand': 
    #             mask = np.random.rand(*total_events.shape[:-1]) < drop_perc
    #             total_events[mask] = 0.0
    #         elif drop_mode == 'fixed':
    #             mask = np.random.rand(*total_events.shape[1:-1]) < drop_perc
    #             total_events[:, mask] = 0.0
    #         return total_events
    #     return drop_token
    def drop_token(self, total_events):
        if self.validation: 
            return total_events
        if self.drop_mode == 'rand': 
            mask = np.random.rand(*total_events.shape[:-1]) < self.drop_perc
            total_events[mask] = 0.0
        elif self.drop_mode == 'fixed':
            mask = np.random.rand(*total_events.shape[1:-1]) < self.drop_perc
            total_events[:, mask] = 0.0
        return total_events
    
    # def get_shift_func(self):
    #     def shift(total_pixels, cropped_shape):
    #         height_diff, width_diff = self.height - cropped_shape[1], self.width - cropped_shape[0]
    #         # print(height_diff, width_diff, cropped_shape, self.height, self.width)
    #         if not self.validation:
    #             new_height_init = np.random.randint(0, height_diff) if height_diff != 0.0 else 0
    #             new_width_init = np.random.randint(0, width_diff)  if width_diff != 0.0 else 0
    #         else:
    #             new_height_init, new_width_init = height_diff // 2, width_diff // 2, 
    #         for i in range(len(total_pixels)): 
    #             total_pixels[i][:, 1] += new_height_init
    #             total_pixels[i][:, 0] += new_width_init
    #         return total_pixels
    #     return shift
    def shift(self, total_pixels, cropped_shape):
        height_diff, width_diff = self.height - cropped_shape[0], self.width - cropped_shape[1]
        # print(height_diff, width_diff, cropped_shape, self.height, self.width)
        if not self.validation:
            new_height_init = np.random.randint(0, height_diff) if height_diff != 0.0 else 0
            new_width_init = np.random.randint(0, width_diff)  if width_diff != 0.0 else 0
        else:
            new_height_init, new_width_init = height_diff // 2, width_diff // 2
            
        # print(1, new_height_init, new_width_init, self.height, self.width, cropped_shape)
        new_height_init -= new_height_init % self.patch_size    #; new_height_init += self.patch_size//2
        new_width_init -= new_width_init % self.patch_size      #; new_width_init += self.patch_size//2
        # print(2, new_height_init, new_width_init)
        
        for i in range(len(total_pixels)): 
            total_pixels[i][:, 0] += new_height_init
            total_pixels[i][:, 1] += new_width_init
        return total_pixels
    
    # def get_crop_to_max_events(self, max_events_per_chunk):
    #     print(' *** get_crop_to_max_events', max_events_per_chunk)
    #     def crop_to_max_events(total_polarity, total_pixels):
    #         if not self.validation:
    #             for i in range(len(total_pixels)): 
    #                 if len(total_polarity[i]) > max_events_per_chunk:
    #                     inds = np.random.choice(list(range(len(total_polarity[i]))), size=max_events_per_chunk, replace=False)
    #                     # print(inds)
    #                     # print('----', total_polarity[i].shape, len(total_polarity[i]), max_events_per_chunk)
    #                     total_polarity[i] = total_polarity[i][inds]
    #                     # print('++++', total_polarity[i].shape)
    #                     total_pixels[i] = total_pixels[i][inds]
    #         return total_polarity, total_pixels
    #     return crop_to_max_events
    def crop_to_max_events(self, total_polarity, total_pixels):
        if not self.validation:
            for i in range(len(total_pixels)): 
                if len(total_polarity[i]) > self.max_events_per_chunk:
                    inds = np.random.choice(list(range(len(total_polarity[i]))), size=self.max_events_per_chunk, replace=False)
                    # print(inds)
                    # print('----', total_polarity[i].shape, len(total_polarity[i]), self.max_events_per_chunk)
                    total_polarity[i] = total_polarity[i][inds]
                    # print('++++', total_polarity[i].shape)
                    total_pixels[i] = total_pixels[i][inds]
        return total_polarity, total_pixels
    

    def __len__(self):
        return len(self.samples)
    
    def get_label_dict(self):
        label_dict = { c:[] for c in set(self.labels) }
        for i,l in enumerate(self.labels): label_dict[l].append(i)
        for k in label_dict: label_dict[k] = torch.IntTensor(label_dict[k])
        return label_dict

    # Return -> [num_timesteps, num_chunk_events, 2pol] | [num_timesteps, num_chunk_events, 2pix_xy], [num_timesteps]
    # def __getitem__(self, idx, return_sparse_array=False):
    def __getitem_v0__(self, idx, return_sparse_array=False):
        
        # print('*********')
        
        filename = self.samples[idx]
        label = self.labels[idx]
        
        # Load sparse matrix
        total_events = pickle.load(open(os.path.join(self.samples_folder + filename), 'rb'))  # events (t x H x W x 2)
        total_events = total_events.todense()
        # print('****** total_events.shape', total_events.shape)
        
        
        # Crop sequence to self.num_sparse_frames
        # if self.crop_in_time: total_events = self.crop_in_time(total_events)
        # if self.crop_in_space: total_events = self.crop_in_space(total_events)
        # if self.drop_token: total_events = self.drop_token(total_events)
        
        if 'max_sample_len_ms' in self.augmentation_params and self.augmentation_params['max_sample_len_ms'] != -1:
            total_events = self.crop_in_time(total_events)
        if 'random_frame_size' in self.augmentation_params and self.augmentation_params['random_frame_size'] is not None:
            total_events = self.crop_in_space(total_events)
        if 'drop_token' in self.augmentation_params and self.augmentation_params['drop_token'][0] != 0.0:
            total_events = self.drop_token(total_events)
        
        # Slice and group into self.chunk_len_ms length
        diff_frames = total_events.shape[0] % self.chunk_size
        if diff_frames != 0 and total_events.shape[0] > self.chunk_size :
            if np.random.rand() < 0: total_events = total_events[:diff_frames]
            else: total_events = total_events[diff_frames:] 
        
        
        # total_chunks = [ total_events[i:i+self.chunk_size].todense() for i in range(0, total_events.shape[0], self.chunk_size) ]    # .sum(0)
        total_chunks = [ total_events[i:i+self.chunk_size] for i in range(0, total_events.shape[0], self.chunk_size) ]    # .sum(0)
        # total_chunks = [ c.sum(0) for c in total_chunks ]
        
        
        if return_sparse_array: return total_chunks
        else:
            total_pixels, total_polarity = [], []
            for nc, c in enumerate(total_chunks):

                bins_init = c.shape[0]; bins_step = bins_init//self.bins
                if bins_step == 0: 
                    print('*****', c.shape, 0, bins_init, bins_step)
                    print(f'Empty chunk [{nc}]')
                    continue

                # c = np.concatenate([ c[i:i+bins_step].sum(0) for i in range(0, bins_init, bins_step) ], axis=2)
                c = np.stack([ c[i:i+bins_step].sum(0) for i in range(0, bins_init, bins_step) ], axis=-1)
                # c = [ c[i:i+bins_step].sum(0) for i in range(0, bins_init, bins_step) ]
                
                if '1' in self.preproc_polarity: c = c.sum(2, keepdims=True)
                
                c = c.reshape(c.shape[0], c.shape[1], c.shape[2]*c.shape[3])
                
                if 'log' in self.preproc_polarity: c = np.log(c + 1)
                elif 'unique' in self.preproc_polarity: c = (c>0).astype('float')
                elif 'norm' in self.preproc_polarity: 
                    raise ValueError('Not implemented')
                    # c = c / c.max(2, keepdims=True)   # [0]
        
        
                polarity = view_as_blocks(c, (self.patch_size,self.patch_size, self.preproc_event_size)); 
                
                # aggregate by pixel (unique), by patch (sum) -> get the ones with >= min_activations | (num_patches, bool)
                inds = (polarity.sum(-1)!=0).reshape(polarity.shape[0], polarity.shape[1], self.patch_size*self.patch_size) \
                    .sum(-1).reshape(polarity.shape[0] * polarity.shape[1]) >= self.min_activations_per_patch
                
                # Reshape to (num_patches x token_dim)
                polarity = polarity.reshape(polarity.shape[0] * polarity.shape[1], self.token_dim)
                
                pixels = np.array([ (i+self.patch_size//2,j+self.patch_size//2) for i in range(0, c.shape[0], self.patch_size) for j in range(0, c.shape[1], self.patch_size) ])
                polarity, pixels = polarity[inds], pixels[inds]
                
                total_pixels.append(torch.tensor(pixels).long()); total_polarity.append(torch.tensor(polarity).long())
    
            # assert len(total_pixels) > 0
            # if self.crop_events: total_polarity, total_pixels = self.crop_events(total_polarity, total_pixels)
            # if self.random_shift: total_pixels = self.random_shift(total_pixels, total_events.shape[1:-1])
            
            if 'crop_to_max_events' in self.augmentation_params and self.augmentation_params['crop_to_max_events'] is not None:
                total_polarity, total_pixels = self.crop_to_max_events(total_polarity, total_pixels)
            if 'random_shift' in self.augmentation_params and self.augmentation_params['random_shift']:
                total_pixels = self.shift(total_pixels, total_events.shape[1:-1])
                
            return total_polarity, total_pixels, label
        
        
        # Return -> [num_timesteps, num_chunk_events, 2pol] | [num_timesteps, num_chunk_events, 2pix_xy], [num_timesteps]
    def __getitem__(self, idx, return_sparse_array=False):
    # def __getitem_v1__(self, idx, return_sparse_array=False):
# %%
        # print(idx)
        # raise ValueError('**********')
        filename = self.samples[idx]
        label = self.labels[idx]
        
        # Load sparse matrix
        total_events = pickle.load(open(os.path.join(self.samples_folder + filename), 'rb'))  # events (t x H x W x 2)
        # total_events = total_events.todense()
        # print('****** total_events.shape', total_events.shape, filename)
        
        ##############################################
        ##############################################
        ##############################################
        # self.chunk_len_ms = 8  # ms
        # self.sparse_frame_len_ms = 2 # each sparse frame is 2 ms
        # self.augmentation_params['max_sample_len_ms'] = 500 # Each sequence must last 500 ms
        # # self.num_sparse_frames = augmentation_params['max_sample_len_ms'] // self.sparse_frame_len_ms
        # self.num_sparse_frames = 250 # 250 sparse frames needed
        # self.chunk_size = 4 # Number of sparse frames needed to complete a chunk
        
        # self.bins = 4
        # self.min_patches_per_chunk = None
        # self.min_events_per_chunk = None
        # self.min_activations_per_patch = int(0.05*self.patch_size*self.patch_size+1)
        ##############################################
        ##############################################
        ##############################################
        
        # Crop sequence to self.num_sparse_frames
        dense = False
        if 'max_sample_len_ms' in self.augmentation_params and self.augmentation_params['max_sample_len_ms'] != -1:
            total_events = self.crop_in_time(total_events)
        if not self.validation and 'rotate' in self.augmentation_params and self.augmentation_params['rotate'] is not None and len(self.augmentation_params['rotate']) > 0:
            total_events = total_events.todense(); dense = True
            angl = np.random.uniform(-self.augmentation_params['rotate']['angle'], self.augmentation_params['rotate']['angle'])
            total_events = ndimage.rotate(total_events, angl, axes=(2,1), reshape=False, mode=self.augmentation_params['rotate']['mode'])
        if 'random_frame_size' in self.augmentation_params and self.augmentation_params['random_frame_size'] is not None:
            total_events = self.crop_in_space(total_events)
            # print('****', total_events.shape)
            
        if not self.validation and self.h_flip and np.random.rand() > 0.5: total_events = total_events[:,:,::-1,:]
        
        # if self.center_frame:
        #     pass
            
        # 1.0. Get chunks by grouping sparse frames
        
        
        total_pixels, total_polarity = [], []
        current_chunk = None
        # sf = total_events[0]
        # for sf_num, sf in enumerate(total_events[::-1]):
        # for sf_num in list(range(len(total_events)-1,-1, -1))[::2]:
            
        # NUM_EXTRA_CHUNKS = 2
        sf_num = len(total_events) - 1
        while sf_num >= 0:
            # print(sf_num)
            
            if current_chunk is None: 
                # print(sf_num-self.chunk_size, sf_num)
                current_chunk = total_events[max(0, sf_num-self.chunk_size):sf_num][::-1]
                if not dense: current_chunk = current_chunk.todense()
                sf_num -= self.chunk_size
                # current_chunk = total_events[min(0, sf_num-self.chunk_size):sf_num][::-1]; sf_num -= self.chunk_size
                if '1' in self.preproc_polarity: current_chunk = current_chunk.sum(-1, keepdims=True)

            else:
                sf = total_events[max(0, sf_num-self.num_extra_chunks):sf_num][::-1]
                if not dense: sf = sf.todense()
                sf_num -= self.num_extra_chunks
                # sf = total_events[min(0, sf_num-self.num_extra_chunks):sf_num][::-1]; sf_num -= self.num_extra_chunks
                if '1' in self.preproc_polarity: sf = sf.sum(-1, keepdims=True)
                current_chunk = np.concatenate([current_chunk, sf])
                

            
            # sf = np.stack([total_events[sf_num], total_events[sf_num-1]]).todense()
            # # print(sf_num)
            
            # # TODO: get 2 sparse frames at the same time
            # # sf = sf.todense()
            # if '1' in self.preproc_polarity: sf = sf.sum(-1, keepdims=True)
                
            # # if current_chunk is None: current_chunk = sf[np.newaxis, ...]
            # # else: current_chunk = np.concatenate([current_chunk, sf[np.newaxis, ...]])
            # if current_chunk is None: current_chunk = sf
            # else: current_chunk = np.concatenate([current_chunk, sf])
            
            # if len(current_chunk) < self.chunk_size: 
            #     # print('aaaaaaaaaaaaaaaa', current_chunk.shape, self.chunk_size, sf_num)
            #     continue
            # else:
                
            if current_chunk.shape[0] >= self.bins:
                
                # print(0, current_chunk.shape)
                
                # Get bins
                bins_init = current_chunk.shape[0];
                bins_step = bins_init//self.bins
                # if bins_step*self.bins < bins_init: bins_step = bins_init//(self.bins-1)
                if bins_step == 0: 
                    print('*****', current_chunk.shape, 0, bins_init, bins_step)
                    # print(f'Empty chunk [{nc}]')
                    # continue
                # chunk_candidate = np.stack([ current_chunk[i:i+bins_step].sum(0) for i in range(0, bins_init, bins_step) ], axis=-1)
                chunk_candidate = []
                for ib_num, i in enumerate(list(range(0, bins_init, bins_step))[:self.bins]):
                    if ib_num == self.bins-1: step = 99999
                    else: step = bins_step
                    chunk_candidate.append(current_chunk[i:i+step].sum(0))
                chunk_candidate = np.stack(chunk_candidate, axis=-1).astype(float)
                # chunk_candidate = np.stack([ current_chunk[i:i+bins_step].sum(0) for i in list(range(0, bins_init, bins_step))[:self.bins] ], axis=-1)
                chunk_candidate = chunk_candidate.reshape(chunk_candidate.shape[0], chunk_candidate.shape[1], chunk_candidate.shape[2]*chunk_candidate.shape[3])
                # print('chunk_candidate.shape', chunk_candidate.shape)
                
                # Extract patches
                # print('1,', sf.shape, current_chunk.shape, chunk_candidate.shape)
                # print('aaa', current_chunk.shape, chunk_candidate.shape, (self.patch_size,self.patch_size, self.preproc_event_size), list(range(0, bins_init, bins_step)))
                polarity = view_as_blocks(chunk_candidate, (self.patch_size,self.patch_size, self.preproc_event_size)); 
                
                # aggregate by pixel (unique), by patch (sum) -> get the ones with >= min_activations | (num_patches, bool)
                inds = (polarity.sum(-1)!=0).reshape(polarity.shape[0], polarity.shape[1], self.patch_size*self.patch_size) \
                    .sum(-1).reshape(polarity.shape[0] * polarity.shape[1]) >= self.min_activations_per_patch
                    
                
                if inds.sum() == 0: continue
                # Check if chunk has the desired patch activations and #events
                if self.min_patches_per_chunk and inds.sum() < self.min_patches_per_chunk: continue
                if self.min_events_per_chunk and polarity.sum() < self.min_events_per_chunk: continue
            
                # break
                # Good chunk -> process and store
            
                # Reshape to (num_patches x token_dim)
                polarity = polarity.reshape(polarity.shape[0] * polarity.shape[1], self.patch_size*self.patch_size*self.preproc_event_size)   # self.token_dim
                
                pixels = np.array([ (i+self.patch_size//2,j+self.patch_size//2) for i in range(0, chunk_candidate.shape[0], self.patch_size) for j in range(0, chunk_candidate.shape[1], self.patch_size) ])
                # pixels = np.array([ (i,j) for i in range((self.patch_size//2), chunk_candidate.shape[1], self.patch_size) for j in range((self.patch_size//2), chunk_candidate.shape[0], self.patch_size) ])
                # print(2, polarity.shape, pixels.shape)
                # print(pixels)
                
                inds = np.where(inds)[0]
                # print(f'{len(total_pixels)} || {inds.shape[0]} patches || {polarity.sum()} events')
                
                # 1.1. Drop patch tokens
                # Apply over the final patch-tokens
                if not self.validation and len(inds)>0 and 'drop_token' in self.augmentation_params and self.augmentation_params['drop_token'][0] != 0.0:
                    # inds = self.drop_token(inds)
                    inds = np.random.choice(inds, replace=False, size=max(1, int(len(inds)*(1-self.augmentation_params['drop_token'][0]))))
                # print('*******', inds.shape, polarity.shape, pixels.shape)
                polarity, pixels = polarity[inds], pixels[inds]

                
                if 'log' in self.preproc_polarity: polarity = np.log(polarity + 1)
                elif 'unique' in self.preproc_polarity: polarity = (polarity>0).astype('float')
                elif 'norm' in self.preproc_polarity: 
                    raise ValueError('Not implemented')
                    # c = c / c.max(2, keepdims=True)   # [0]
                    
                assert len(pixels) > 0 and len(polarity) > 0
                # 2.0. Process token_dim
                total_polarity.append(torch.tensor(polarity))
                total_pixels.append(torch.tensor(pixels).long())
                current_chunk = None
                
                
        # Ensure at least one chunk in the list
        if len(total_pixels) == 0 and current_chunk.shape[0] >= self.bins:
            # Get bins
            bins_init = current_chunk.shape[0];
            bins_step = bins_init//self.bins
            # if bins_step*self.bins < bins_init: bins_step = bins_init//(self.bins-1)
            if bins_step == 0: 
                print('*****', current_chunk.shape, 0, bins_init, bins_step)
                # print(f'Empty chunk [{nc}]')
                # continue
            # chunk_candidate = np.stack([ current_chunk[i:i+bins_step].sum(0) for i in range(0, bins_init, bins_step) ], axis=-1)
            chunk_candidate = []
            for ib_num, i in enumerate(list(range(0, bins_init, bins_step))[:self.bins]):
                if ib_num == self.bins-1: step = 99999
                else: step = bins_step
                chunk_candidate.append(current_chunk[i:i+step].sum(0))
            chunk_candidate = np.stack(chunk_candidate, axis=-1).astype(float)
            # chunk_candidate = np.stack([ current_chunk[i:i+bins_step].sum(0) for i in list(range(0, bins_init, bins_step))[:self.bins] ], axis=-1)
            chunk_candidate = chunk_candidate.reshape(chunk_candidate.shape[0], chunk_candidate.shape[1], chunk_candidate.shape[2]*chunk_candidate.shape[3])
                
                
            # Extract patches
            # print('1,', sf.shape, current_chunk.shape, chunk_candidate.shape)
            polarity = view_as_blocks(chunk_candidate, (self.patch_size,self.patch_size, self.preproc_event_size)); 
            
            # aggregate by pixel (unique), by patch (sum) -> get the ones with >= min_activations | (num_patches, bool)
            inds = (polarity.sum(-1)!=0).reshape(polarity.shape[0], polarity.shape[1], self.patch_size*self.patch_size) \
                .sum(-1).reshape(polarity.shape[0] * polarity.shape[1]) >= self.min_activations_per_patch
            if inds.sum() == 0:
                inds = (polarity.sum(-1)!=0).reshape(polarity.shape[0], polarity.shape[1], self.patch_size*self.patch_size) \
                    .sum(-1).reshape(polarity.shape[0] * polarity.shape[1]) >= 1.0
                if inds.sum() == 0:
                    inds = (polarity.sum(-1)!=0).reshape(polarity.shape[0], polarity.shape[1], self.patch_size*self.patch_size) \
                        .sum(-1).reshape(polarity.shape[0] * polarity.shape[1]) >= 0.0
            if inds.sum() == 0: 
                print(len(inds), inds.sum(), current_chunk.shape, chunk_candidate.shape, sf_num, len(total_polarity), len(total_pixels), total_events.shape, filename)
                
            
            # break
            # Good chunk -> process and store
        
            # Reshape to (num_patches x token_dim)
            polarity = polarity.reshape(polarity.shape[0] * polarity.shape[1], self.patch_size*self.patch_size*self.preproc_event_size)   # self.token_dim
            
            pixels = np.array([ (i+self.patch_size//2,j+self.patch_size//2) for i in range(0, chunk_candidate.shape[0], self.patch_size) for j in range(0, chunk_candidate.shape[1], self.patch_size) ])
            # print(2, polarity.shape, pixels.shape)
            
            inds = np.where(inds)[0]
            # print(f'{len(total_pixels)} || {inds.shape[0]} patches || {polarity.sum()} events')
            
            # 1.1. Drop patch tokens
            # Apply over the final patch-tokens
            if not self.validation and len(inds)>0 and 'drop_token' in self.augmentation_params and self.augmentation_params['drop_token'][0] != 0.0:
                # inds = self.drop_token(inds)
                inds = np.random.choice(inds, replace=False, size=max(1, int(len(inds)*(1-self.augmentation_params['drop_token'][0]))))
            # print('*******', inds.shape, polarity.shape, pixels.shape)
            polarity, pixels = polarity[inds], pixels[inds]

            
            if 'log' in self.preproc_polarity: polarity = np.log(polarity + 1)
            elif 'unique' in self.preproc_polarity: polarity = (polarity>0).astype('float')
            elif 'norm' in self.preproc_polarity: 
                raise ValueError('Not implemented')
                # c = c / c.max(2, keepdims=True)   # [0]
                
            # 2.0. Process token_dim
            assert len(pixels) > 0 and len(polarity) > 0
            total_polarity.append(torch.tensor(polarity))
            total_pixels.append(torch.tensor(pixels).long())
            current_chunk = None
            
            
        if 'random_shift' in self.augmentation_params and self.augmentation_params['random_shift']:
            total_pixels = self.shift(total_pixels, total_events.shape[1:-1])
        
        # assert len(total_polarity) > 0 and len(total_pixels) > 0, f'{len(total_pixels)} , {len(total_polarity)} , {current_chunk.shape} , {self.bins} , {total_events.shape} , {filename}'
        return total_polarity, total_pixels, label
            

    

# Return the batch sample indices randomly.
class CustomBatchSampler():
    
    # TODO: remove used samples from dict and re-create when empty ?
    # - TODO: add samples_per_class
    def __init__(self, batch_size, label_dict, sample_repetitions=1, drop_last=False):
        
        assert batch_size % sample_repetitions == 0
        self.batch_size = batch_size
        self.label_dict = label_dict
        self.sample_repetitions = sample_repetitions
        self.drop_last = drop_last
        self.generator = torch.Generator()
        self.generator.manual_seed(0)
        self.num_classes = len(self.label_dict)
        self.unique_labels = list(self.label_dict.keys())
        
    def __len__(self):
        epoch_length = sum([ len(v) for v in self.label_dict.values() ])*self.sample_repetitions // self.batch_size
        print('**********', epoch_length)
        return epoch_length
        # return 5
    
    def __iter__(self):
        
        # if self.label_dict is not None: 
            # keys = list(self.label_dict.keys())
            # num_classes = len(keys)
            
        total_labels = []
        while True:
            # ks = []
            inds = []
            
            for b in range(self.batch_size // self.sample_repetitions):
                
                if len(total_labels) == 0: total_labels = self.unique_labels.copy()
                
                k = np.random.randint(0, len(total_labels), size=(1))[0]
                k = total_labels.pop(k)
                # ks.append(k)
                
                # if self.sample_dict is not None:
                # k = torch.randint(0, self.num_classes, size=(1,), generator=self.generator)[0].item()
                # k = np.random.randint(0, self.num_classes, size=(1))[0]
                # k = keys[torch.randint(0, num_classes, size=(1,), generator=self.generator)[0].item()]
                num_k_samples = len(self.label_dict[k])
                # ind = torch.randint(0, num_k_samples, size=(1,), generator=self.generator)[0].item()
                ind = np.random.randint(0, num_k_samples, size=(1))[0]
                ind = self.label_dict[k][ind]
                for _ in range(self.sample_repetitions):  inds.append(ind)
            
            # print('***', len(ks), ks)
            yield inds


# Pad sequences by timesteps and events
# Samples: ([batch_size], [timesteps/chunk], [events], event_data)
def pad_list_of_sequences(samples, token_size, pre_padding = True):
    # print(f'time_steps || max {max([ len(s) for s in samples ])} | mean {np.mean([ len(s) for s in samples ])}')
    # print(f'events_num || max {max([ chunk.shape[0] for sample in samples for chunk in sample ])} | mean {np.mean([ chunk.shape[0] for sample in samples for chunk in sample ])}')
    max_timesteps = max([ len(s) for s in samples ])
    batch_size = len(samples)
    max_event_num = max([ chunk.shape[0] for sample in samples for chunk in sample ])
    # token_size = samples[0][0][0].shape[-1]
    
    batch_data = torch.zeros(max_timesteps, batch_size, max_event_num, token_size)
    for num_sample, action_sample in enumerate(samples):
        num_chunks = len(action_sample)
        # print([ c.shape for c in action_sample ])
        for chunk_num, chunk in enumerate(action_sample):
            chunk_events = chunk.shape[0]
            # print(' ** {} {}'.format(batch_data.shape, chunk_events))
            if chunk_events == 0:
                # pass
                continue
            if pre_padding: batch_data[-(num_chunks-chunk_num), num_sample, -chunk_events:, :] = chunk
            else: batch_data[chunk_num, num_sample, :chunk_events, :] = chunk
            
    return batch_data
            

# def get_custom_collate_fn(pre_padding = True):
#     def custom_collate_fn(batch_samples):
#         pols, pixels, labels = [], [], []
#         for sample in batch_samples:
#             pols.append(sample[0])
#             pixels.append(sample[1])
#             labels.append(sample[2])
            
#         token_size = pols[0][0].shape[-1]
            
#         pols = pad_list_of_sequences(pols, token_size, pre_padding)
#         pixels = pad_list_of_sequences(pixels, 2, pre_padding)
#         # if '1' in preproc_polarity: pols = pols.sum(-1, keepdims=True)
#         # if 'log' in preproc_polarity: pols = torch.log(pols + 1)
#         # elif 'unique' in preproc_polarity: pols = (pols>0).float()
#         # elif 'norm' in preproc_polarity: pols = pols / pols.max(2, True)[0]
        
#         pols, pixels, labels = pols, pixels.long(), torch.tensor(labels).long()
#         # print('+++++', pols.shape, pixels.shape, labels.shape)
#         return pols, pixels, labels
    # return custom_collate_fn



class DVS128DataModule(LightningDataModule):
    def __init__(self, batch_size, chunk_len_ms, 
                 patch_size, min_activations_per_patch, bins,
                 min_patches_per_chunk, min_events_per_chunk, num_extra_chunks,
                 augmentation_params, 
                 dataset_name,
                 from_frames=True, 
                 skip_last_event=False, sample_repetitions=1, preproc_polarity=None, 
                 one_sample_per_chunk=False,
                 custom_sampler = True,
                 workers=8, pin_memory=False, classes_to_exclude=[], balance=None):
        super().__init__()
        self.batch_size = batch_size
        # self.event_size = event_size
        self.chunk_len_ms = chunk_len_ms
        self.patch_size = patch_size
        self.min_activations_per_patch = min_activations_per_patch
        self.bins = bins
        self.min_patches_per_chunk = min_patches_per_chunk
        self.min_events_per_chunk = min_events_per_chunk
        self.num_extra_chunks = num_extra_chunks
        
        
        self.augmentation_params = augmentation_params
        self.dataset_name = dataset_name
        self.from_frames = from_frames
        self.workers = workers
        self.sample_repetitions = sample_repetitions
        self.preproc_polarity = preproc_polarity
        self.skip_last_event = skip_last_event
        self.pin_memory = pin_memory
        self.classes_to_exclude = classes_to_exclude
        
        self.pre_padding = True
        self.one_sample_per_chunk = one_sample_per_chunk
        self.custom_sampler = custom_sampler
        
        
        
        assert self.chunk_len_ms / 2 % self.bins == 0.0
        if not from_frames: raise ValueError('not from_frames not implemented')
        
        self.dataset_name = dataset_name
        if dataset_name == 'DVS128':
            # if self.chunk_len_ms == 8 or self.bins not in [1,2]: self.data_folder = './datasets/DvsGesture/clean_dataset_frames_2000/'
            # # self.data_folder = './datasets/DvsGesture/clean_dataset_frames_6000/'
            # else: self.data_folder = './datasets/DvsGesture/clean_dataset_frames_12000/'
            self.data_folder = './datasets/DvsGesture/clean_dataset_frames_2000/'

            self.width, self.height = 128, 128
            self.num_classes = 12 - len(classes_to_exclude)
            self.class_mapping = copy.deepcopy(class_mapping)
            for c in classes_to_exclude: del self.class_mapping[c]
            # self.class_mapping = { i:l for i,_,l in enumerate(sorted(self.class_mapping.items(), key=lambda x:x[0])) }
            self.class_mapping = { i:l[1] for i,l in enumerate(sorted(self.class_mapping.items(), key=lambda x:x[0])) }
        elif dataset_name == 'ASL_DVS':
            self.data_folder = './datasets/ICCV2019_DVS_dataset/clean_dataset_frames_2000/'
            self.width, self.height = 240, 180
            # self.width, self.height = 180, 240
            self.num_classes = 24
            self.class_mapping = { i:l for i,l in enumerate('a b c d e f g h i k l m n o p q r s t u v w x y'.split()) }
        elif dataset_name == 'HMDB':
            self.data_folder = './datasets/HMDB_DVS/dataset_s0.3_2000/'
            self.width, self.height = 240, 180
            # self.width, self.height = 180, 240
            self.num_classes = 51
            self.class_mapping = { i:l for i,l in enumerate(range(self.num_classes)) }
        elif dataset_name == 'UCF101':
            self.data_folder = './datasets/UCF101_DVS/dataset_split1_2000/'
            self.width, self.height = 240, 180
            # self.width, self.height = 180, 240
            self.num_classes = 101
            self.class_mapping = { i:l for i,l in enumerate(range(self.num_classes)) }
        elif dataset_name == 'UCF50':
            self.data_folder = './datasets/UCF50_DVS/dataset_split1_2000/'
            self.width, self.height = 240, 180
            # self.width, self.height = 180, 240
            self.num_classes = 50
            self.class_mapping = { i:l for i,l in enumerate(range(self.num_classes)) }
        elif dataset_name == 'SLAnimals_3s':
            # if self.chunk_len_ms == 8 or self.bins not in [1,2]: self.data_folder = './datasets/SL_animal_splits/dataset_3sets_2000/'
            # else: self.data_folder = './datasets/SL_animal_splits/dataset_3sets_12000/'
            self.data_folder = './datasets/SL_animal_splits/dataset_3sets_2000/'
            self.width, self.height = 128, 128
            # self.width, self.height = 180, 240
            self.num_classes = 19
            self.class_mapping = { i:l for i,l in enumerate(range(self.num_classes)) }
        elif dataset_name == 'SLAnimals_4s':
            # if self.chunk_len_ms == 8 or self.bins not in [1,2]: self.data_folder = './datasets/SL_animal_splits/dataset_4sets_2000/'
            # else: self.data_folder = './datasets/SL_animal_splits/dataset_4sets_12000/'
            self.data_folder = './datasets/SL_animal_splits/dataset_4sets_2000/'
            self.width, self.height = 128, 128
            # self.width, self.height = 180, 240
            self.num_classes = 19
            self.class_mapping = { i:l for i,l in enumerate(range(self.num_classes)) }
        elif dataset_name == 'N_Cars':
            self.data_folder = './datasets/Prophesee_Dataset_n_cars/dataset_2000/'
            self.width, self.height = 128, 128
            # self.width, self.height = 180, 240
            self.num_classes = 2
            self.class_mapping = { i:l for i,l in enumerate(range(self.num_classes)) }
        elif dataset_name == 'Caltech':
            self.data_folder = './datasets/N_Caltech_101/clean_dataset_frames_2000/'
            self.width, self.height = 240, 180
            # self.width, self.height = 180, 240
            self.num_classes = 101
            self.class_mapping = { i:l for i,l in enumerate(range(self.num_classes)) }
        else: raise ValueError(f'Dataset [{dataset_name}] not handled')
        
        # if from_frames: self.data_folder = './datasets/DvsGesture/clean_dataset_frames_2000/'
        # else: self.data_folder = './datasets/DvsGesture/clean_dataset/'

        self.preproc_event_size = 1 if '1' in self.preproc_polarity else 2
        self.preproc_event_size *= bins
        self.token_dim = patch_size*patch_size * self.preproc_event_size


    def custom_collate_fn(self, batch_samples):
        pols, pixels, labels = [], [], []
        
        # one_sample_per_chunk | Each chunk will be a sample with 1 time-step
        if self.one_sample_per_chunk: sample_id, sample_labels, idx = [], [], 0
        
        for num_sample, sample in enumerate(batch_samples): 
            # Sample -> time_sequence
            # #samples == batch_size
            
            if sample is None or len(sample[0]) == 0: 
                print('Empty sample')
                print(len(sample), len(sample[0]))
                continue
            
            # print(len(sample[0]), sample[0][0].shape, len(sample[1]), sample[1][0].shape)
            if self.one_sample_per_chunk:
                for i in range(len(sample[0])):
                    pols.append([sample[0][i]])
                    pixels.append([sample[1][i]])
                    labels.append(sample[2])
                    # sample_id.append(num_sample)
                sample_id.append(list(range(idx, idx+len(sample[0]))))
                sample_labels.append(sample[2])
                idx += len(sample[0])
            else:
                pols.append(sample[0])
                pixels.append(sample[1])
                labels.append(sample[2])
            
        token_size = pols[0][0].shape[-1]
            
        pols = pad_list_of_sequences(pols, token_size, self.pre_padding)
        pixels = pad_list_of_sequences(pixels, 2, self.pre_padding)
        # if '1' in preproc_polarity: pols = pols.sum(-1, keepdims=True)
        # if 'log' in preproc_polarity: pols = torch.log(pols + 1)
        # elif 'unique' in preproc_polarity: pols = (pols>0).float()
        # elif 'norm' in preproc_polarity: pols = pols / pols.max(2, True)[0]
        
        pols, pixels, labels = pols, pixels.long(), torch.tensor(labels).long()
        # print('+++++', pols.shape, pixels.shape, labels.shape)
        if self.one_sample_per_chunk: 
            sample_labels = torch.tensor(sample_labels).long()
            return pols, pixels, (labels, sample_id, sample_labels)
        else: return pols, pixels, labels
    
    
    def train_dataloader(self):
        # if self.from_frames:
        dt = DVS128Dataset_from_frames(self.data_folder+'train/', chunk_len_ms = self.chunk_len_ms, 
                           validation=False, 
                           preproc_polarity=self.preproc_polarity, patch_size=self.patch_size,
                           min_activations_per_patch=self.min_activations_per_patch,
                           bins = self.bins, 
                           min_patches_per_chunk = self.min_patches_per_chunk, min_events_per_chunk = self.min_events_per_chunk,
                           num_extra_chunks = self.num_extra_chunks,
                           dataset_name=self.dataset_name, height=self.height, width=self.width,
                           augmentation_params=self.augmentation_params, 
                           classes_to_exclude=self.classes_to_exclude)
        # else:
        #     dt = DVS128Dataset(self.data_folder+'train/', chunk_len_ms = self.chunk_len_ms, 
        #                    skip_last_event=self.skip_last_event, classes_to_exclude=self.classes_to_exclude)
        if self.custom_sampler: 
            sampler = CustomBatchSampler(batch_size=self.batch_size, label_dict=dt.get_label_dict(), sample_repetitions=self.sample_repetitions)
            dl = DataLoader(dt, batch_sampler=sampler, collate_fn=self.custom_collate_fn, num_workers=self.workers, pin_memory=self.pin_memory)
        else:
            dl = DataLoader(dt, batch_size=self.batch_size, collate_fn=self.custom_collate_fn, shuffle=True, num_workers=self.workers, pin_memory=self.pin_memory)
        return dl
    def val_dataloader(self):
        # if self.from_frames:
        dt = DVS128Dataset_from_frames(self.data_folder+'test/', chunk_len_ms = self.chunk_len_ms, 
                           validation=True, 
                           preproc_polarity=self.preproc_polarity, patch_size=self.patch_size,
                           min_activations_per_patch=self.min_activations_per_patch,
                           bins = self.bins, 
                           min_patches_per_chunk = self.min_patches_per_chunk, min_events_per_chunk = self.min_patches_per_chunk,
                           num_extra_chunks = self.num_extra_chunks,
                           dataset_name=self.dataset_name, height=self.height, width=self.width,
                           augmentation_params=self.augmentation_params, 
                           classes_to_exclude=self.classes_to_exclude)
        # else:
        #     dt = DVS128Dataset(self.data_folder+'test/', chunk_len_ms = self.chunk_len_ms, 
        #                    skip_last_event=self.skip_last_event, classes_to_exclude=self.classes_to_exclude)
        # sampler = CustomBatchSampler(batch_size=self.batch_size, label_dict=dt.get_label_dict(), sample_repetitions=self.sample_repetitions)
        # dl = DataLoader(dt, batch_sampler=sampler, collate_fn=get_custom_collate_fn(pre_padding=True), num_workers=self.workers)
        # dl = DataLoader(dt, batch_size=(self.batch_size//2)+1, collate_fn=get_custom_collate_fn(pre_padding=True), num_workers=self.workers, pin_memory=self.pin_memory)
        dl = DataLoader(dt, batch_size=(self.batch_size//2)+1, shuffle=False, collate_fn=self.custom_collate_fn, num_workers=self.workers, pin_memory=self.pin_memory)
        return dl
    
    


# %%

if __name__ == '__main__':
    
    from tqdm import tqdm
    import os
    os.chdir('../..')
    import time
    
    np.random.seed(0)
    

    data_params = {'batch_size': 32, 'sample_repetitions': 1, 'chunk_len_ms': 20, 
                'classes_to_exclude':[], 'workers': 1,
                'preproc_polarity': 'log', 'pin_memory': False, 
                'patch_size': 8, 'min_activations_per_patch': 0.1,
                'bins': 1,
                'min_patches_per_chunk': 128, 'min_events_per_chunk': None, 'num_extra_chunks': 8,
                'one_sample_per_chunk': False, 
                # 'one_sample_per_chunk': True, 
                # 'event_size': 2,
                # 'preproc_polarity': None,
                # 'dataset_name': 'DVS128',
                # 'dataset_name': 'ASL_DVS',
                'dataset_name': 'HMDB',
                # 'dataset_name': 'Caltech',
                'augmentation_params': { 'max_sample_len_ms': 200, 'random_frame_size': 0.75,
                           'random_shift': True, 
                            'drop_token': (0.2, 'fixed'), 
                           # 'drop_token': (0.0, 'fixed'), 
                           'crop_to_max_events': None,
                           } }
    # validation = True
    validation = False
    
    # from_frames = False
    # dm = DVS128DataModule(**data_params, from_frames = from_frames)
    # if not validation: dlf = dm.train_dataloader()
    # else: dlf = dm.val_dataloader()

    from_frames = True
    dm = DVS128DataModule(**data_params, from_frames = from_frames)
    
    # Initialize self for testing
    dlt = dm.val_dataloader()
    dt = dlt.dataset
    self = dt
    batch_samples = [dt[0]]
    idx = 32
    filename = self.samples[idx]
    label = self.labels[idx]
        
        
    if not validation: dlt = dm.train_dataloader()
    else: dlt = dm.val_dataloader()
    
    print(dlt.dataset.num_classes)      #dlf.dataset.num_classes, 
    
    # dlt = iter(dlt)
    # dlf = iter(dlf)
    
    # for _ in tqdm(range(dlf.__len__())): 
    # for _ in tqdm(range(100)): 
    # for i in range(800): 
    #     batch_data_t = next(dlt)
    t = time.time()
    times = []
    print(' * time-steps x batch_size x num_patches x token_dim *')
    for i, batch_data_t in enumerate(dlt): 
        # print(i, batch_data_f[0].shape, batch_data_t[0].shape, batch_data_f[2], batch_data_t[2])
        # print(i, batch_data_t[0].shape, batch_data_t[1].shape, batch_data_t[2])
        print(i, batch_data_t[0].shape, batch_data_t[1].shape, batch_data_t[2].shape)
        t = time.time()-t
        times.append(t)
        print(np.mean(times), t)
        t = time.time()
        
        # del batch_data_t

        # break
    
    
    
# %%

    if False:
        # %%
        dlt = dm.val_dataloader()
        dt = dlt.dataset
        self = dt
        batch_samples = [dt[0]]


        # %%
        
        idx = 32
        filename = self.samples[idx]
        label = self.labels[idx]
        
        # %%
        
        label_dict = dt.get_label_dict()
        num_samples = sum([ len(v) for v in label_dict.values() ])
        max_len = max([ len(v) for v in label_dict.values() ])
        class_weigths = { k:max_len/len(v) for k,v in label_dict.items() }
        class_weigths = { k:(len(v), num_samples/(len(label_dict)*len(v))) for k,v in label_dict.items() }
       




