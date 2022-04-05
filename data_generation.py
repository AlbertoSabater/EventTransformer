from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch

import os
import pickle
import numpy as np
import json
from skimage.util import view_as_blocks
import copy
from scipy import ndimage


DVS128_class_mapping = {0: 'background', 1: 'hand_clapping', 2: 'right_hand_wave', 
                 3: 'left_hand_wave', 4: 'right_arm_clockwise', 
                 5: 'right_arm_counter_clockwise', 6: 'left_arm_clockwise', 
                 7: 'left_arm_counter_clockwise', 8: 'arm_roll', 
                 9: 'air_drums', 10: 'air_guitar', 11: 'other_gestures'}



# EventDataset load the event data organized in sparse frames
# Each sparse frame represents 2-12ms
# Contiguous frames are processed together to represent a time-window (24-48-100ms)
class EventDataset(Dataset):
    def __init__(self, samples_folder,  # Folder from where to load the data
                 chunk_len_ms,          # Base time-window length (ms)
                 validation,            # Wether to use data augmentation or not
                 augmentation_params,   # Augmentation params
                 preproc_polarity,      # '1' to merge both polarities
                 patch_size,            # Side of the patches
                 bins,                  # Histogram bins
                 min_activations_per_patch,     # Percentage of minimum required events per activated patch
                 min_patches_per_chunk, # Minimum amount of patches per time-window
                 num_extra_chunks,      # Number of extra sparse frames to extend the time-window if needed
                 dataset_name, height, width, 
                 classes_to_exclude=[]  # Used to exclude classes from DV128
                 ):
        
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
        self.num_extra_chunks = num_extra_chunks
        
        # Define data augmentation functions
        self.augmentation_params = augmentation_params
        if augmentation_params is not None and len(augmentation_params) != 0:
            # Size of the cropped event-sequences
            if 'max_sample_len_ms' in augmentation_params and augmentation_params['max_sample_len_ms'] != -1:
                self.num_sparse_frames = augmentation_params['max_sample_len_ms'] // self.sparse_frame_len_ms
            # Minimum values of cropped samples
            if 'random_frame_size' in augmentation_params and augmentation_params['random_frame_size'] is not None:
                self.x_lims = (int(width*augmentation_params['random_frame_size']), width)
                self.y_lims = (int(height*augmentation_params['random_frame_size']), height)
            # Percentage of tokens to drop during training
            if 'drop_token' in augmentation_params and augmentation_params['drop_token'][0] != 0.0:
                self.drop_perc, self.drop_mode = augmentation_params['drop_token']
            self.h_flip = augmentation_params.get('h_flip', False)
                
        
        self.bins = bins
        self.preproc_polarity = preproc_polarity
        self.patch_size = patch_size
        self.original_event_size = 1 if '1' in self.preproc_polarity else 2
        self.preproc_event_size = self.original_event_size*bins
        self.token_dim = patch_size*patch_size * self.preproc_event_size
        
        if min_activations_per_patch > 0 and min_activations_per_patch <= 1: 
            self.min_activations_per_patch = int(min_activations_per_patch*patch_size*patch_size+1)
        else: self.min_activations_per_patch = 0
        
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
            
        
    # Weigths to balance datasets
    def get_class_weights(self):
        label_dict = self.get_label_dict()
        label_dict = { k:label_dict[k] for k in sorted(label_dict) }
        num_samples = sum([ len(v) for v in label_dict.values() ])
        class_weigths = [ num_samples/(len(label_dict)*len(v)) for k,v in label_dict.items() ]
        return torch.tensor(class_weigths)
             
    
    # Crop the given event-streams in time
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
        
    
    # Crop event-streams in space
    def crop_in_space(self, total_events):
        _, y_size, x_size, _ = total_events.shape
        if not self.validation:     # Crop sequence randomly
            new_x_size = np.random.randint(self.x_lims[0], self.x_lims[1]+1)
            new_y_size = np.random.randint(self.y_lims[0], self.y_lims[1]+1)
            
            if self.patch_size != 1:
                new_x_size -= new_x_size % self.patch_size
                new_y_size -= new_y_size % self.patch_size
            
            x_init = np.random.randint(x_size - new_x_size+1); x_end = x_init + new_x_size
            y_init = np.random.randint(y_size - new_y_size+1); y_end = y_init + new_y_size
            total_events = total_events[:, y_init:y_end, x_init:x_end, :]
        else:                       # Crop to the middle part
            new_x_size = (self.x_lims[0] + self.x_lims[1])//2
            new_y_size = (self.y_lims[0] + self.y_lims[1])//2
            
            if self.patch_size != 1:
                new_x_size -= new_x_size % self.patch_size
                new_y_size -= new_y_size % self.patch_size
                
            x_init = (x_size - new_x_size)//2; x_end = x_init + new_x_size
            y_init = (y_size - new_y_size)//2; y_end = y_init + new_y_size
            total_events = total_events[:, y_init:y_end, x_init:x_end, :]
        assert total_events.shape[1] == new_y_size and total_events.shape[2] == new_x_size, print(total_events.shape, new_y_size, new_x_size)
        return total_events
        
        
    
    # Remove random events from sequence based on percentage
    # drop_mode == 'fixed' -> drop same pixels for all the sequence
    # drop_mode == 'rand' -> drop random events in each time-step
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
    
    # Shift patches in space
    def shift(self, total_pixels, cropped_shape):
        height_diff, width_diff = self.height - cropped_shape[0], self.width - cropped_shape[1]
        if not self.validation:
            new_height_init = np.random.randint(0, height_diff) if height_diff != 0.0 else 0
            new_width_init = np.random.randint(0, width_diff)  if width_diff != 0.0 else 0
        else:
            new_height_init, new_width_init = height_diff // 2, width_diff // 2
            
        new_height_init -= new_height_init % self.patch_size    #; new_height_init += self.patch_size//2
        new_width_init -= new_width_init % self.patch_size      #; new_width_init += self.patch_size//2
        
        for i in range(len(total_pixels)): 
            total_pixels[i][:, 0] += new_height_init
            total_pixels[i][:, 1] += new_width_init
        return total_pixels
    

    def __len__(self):
        return len(self.samples)
    
    def get_label_dict(self):
        label_dict = { c:[] for c in set(self.labels) }
        for i,l in enumerate(self.labels): label_dict[l].append(i)
        for k in label_dict: label_dict[k] = torch.IntTensor(label_dict[k])
        return label_dict

        
    # Return -> [num_timesteps, num_chunk_events, 2pol], [num_timesteps, num_chunk_events, 2pix_xy], [num_timesteps]
    def __getitem__(self, idx, return_sparse_array=False):

        filename = self.samples[idx]
        label = self.labels[idx]
        
        # Load sparse matrix
        total_events = pickle.load(open(os.path.join(self.samples_folder + filename), 'rb'))  # events (t x H x W x 2)
        
        # Crop sequence to self.num_sparse_frames
        if 'max_sample_len_ms' in self.augmentation_params and self.augmentation_params['max_sample_len_ms'] != -1:
            total_events = self.crop_in_time(total_events)
        if 'random_frame_size' in self.augmentation_params and self.augmentation_params['random_frame_size'] is not None:
            total_events = self.crop_in_space(total_events)
            
        if not self.validation and self.h_flip and np.random.rand() > 0.5: total_events = total_events[:,:,::-1,:]
        
            
        total_pixels, total_polarity = [], []
        current_chunk = None

        # Iterate until read all the total_events (max_sample_len_ms)
        sf_num = len(total_events) - 1
        while sf_num >= 0:

            # Get chunks by grouping sparse frames
            if current_chunk is None: 
                current_chunk = total_events[max(0, sf_num-self.chunk_size):sf_num][::-1]
                current_chunk = current_chunk.todense()
                sf_num -= self.chunk_size
                if '1' in self.preproc_polarity: current_chunk = current_chunk.sum(-1, keepdims=True)
            else:
                sf = total_events[max(0, sf_num-self.num_extra_chunks):sf_num][::-1]
                sf = sf.todense()
                sf_num -= self.num_extra_chunks
                if '1' in self.preproc_polarity: sf = sf.sum(-1, keepdims=True)
                current_chunk = np.concatenate([current_chunk, sf])
                

            if current_chunk.shape[0] < self.bins: continue
            # if current_chunk.shape[0] >= self.bins:
            # Divide time-window into bins
            bins_init = current_chunk.shape[0];
            bins_step = bins_init//self.bins
            chunk_candidate = []
            for ib_num, i in enumerate(list(range(0, bins_init, bins_step))[:self.bins]):
                if ib_num == self.bins-1: step = 99999
                else: step = bins_step
                chunk_candidate.append(current_chunk[i:i+step].sum(0))
            chunk_candidate = np.stack(chunk_candidate, axis=-1).astype(float)
            chunk_candidate = chunk_candidate.reshape(chunk_candidate.shape[0], chunk_candidate.shape[1], chunk_candidate.shape[2]*chunk_candidate.shape[3])
            
            # Extract patches
            polarity = view_as_blocks(chunk_candidate, (self.patch_size,self.patch_size, self.preproc_event_size)); 
            # aggregate by pixel (unique), by patch (sum) -> get the ones with >= min_activations | (num_patches, bool)
            inds = (polarity.sum(-1)!=0).reshape(polarity.shape[0], polarity.shape[1], self.patch_size*self.patch_size) \
                .sum(-1).reshape(polarity.shape[0] * polarity.shape[1]) >= self.min_activations_per_patch
            
            if inds.sum() == 0: continue
            # Check if chunk has the desired patch activations and #events
            if self.min_patches_per_chunk and inds.sum() < self.min_patches_per_chunk: continue
        
            # Reshape to (num_patches x token_dim)
            polarity = polarity.reshape(polarity.shape[0] * polarity.shape[1], self.patch_size*self.patch_size*self.preproc_event_size)   # self.token_dim
            # Get pixel locations
            pixels = np.array([ (i+self.patch_size//2,j+self.patch_size//2) for i in range(0, chunk_candidate.shape[0], self.patch_size) for j in range(0, chunk_candidate.shape[1], self.patch_size) ])
            
            inds = np.where(inds)[0]
            
            # Drop patch tokens
            # Apply over the final patch-tokens
            if not self.validation and len(inds)>0 and 'drop_token' in self.augmentation_params and self.augmentation_params['drop_token'][0] != 0.0:
                inds = np.random.choice(inds, replace=False, size=max(1, int(len(inds)*(1-self.augmentation_params['drop_token'][0]))))
            polarity, pixels = polarity[inds], pixels[inds]

            if 'log' in self.preproc_polarity: polarity = np.log(polarity + 1)
            else: raise ValueError('Not implemented', self.preproc_polarity)
                
            assert len(pixels) > 0 and len(polarity) > 0
            total_polarity.append(torch.tensor(polarity))
            total_pixels.append(torch.tensor(pixels).long())
            current_chunk = None


        if 'random_shift' in self.augmentation_params and self.augmentation_params['random_shift']:
            total_pixels = self.shift(total_pixels, total_events.shape[1:-1])
        
        return total_polarity, total_pixels, label
            

    

# Return the batch sample indices randomly.
class CustomBatchSampler():
    
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
        return epoch_length
    
    def __iter__(self):
        
        total_labels = []
        while True:
            inds = []
            for b in range(self.batch_size // self.sample_repetitions):
                if len(total_labels) == 0: total_labels = self.unique_labels.copy()
                k = np.random.randint(0, len(total_labels), size=(1))[0]
                k = total_labels.pop(k)
                num_k_samples = len(self.label_dict[k])
                ind = np.random.randint(0, num_k_samples, size=(1))[0]
                ind = self.label_dict[k][ind]
                for _ in range(self.sample_repetitions):  inds.append(ind)
            
            yield inds


# Pad sequences by timesteps and #events
# Samples: ([batch_size], [timesteps/chunk], [events], event_data)
def pad_list_of_sequences(samples, token_size, pre_padding = True):
    max_timesteps = max([ len(s) for s in samples ])
    batch_size = len(samples)
    max_event_num = max([ chunk.shape[0] for sample in samples for chunk in sample ])
    
    batch_data = torch.zeros(max_timesteps, batch_size, max_event_num, token_size)
    for num_sample, action_sample in enumerate(samples):
        num_chunks = len(action_sample)
        for chunk_num, chunk in enumerate(action_sample):
            chunk_events = chunk.shape[0]
            if chunk_events == 0:
                continue
            if pre_padding: batch_data[-(num_chunks-chunk_num), num_sample, -chunk_events:, :] = chunk
            else: batch_data[chunk_num, num_sample, :chunk_events, :] = chunk
            
    return batch_data
            


class Event_DataModule(LightningDataModule):
    def __init__(self, batch_size, chunk_len_ms, 
                 patch_size, min_activations_per_patch, bins,
                 min_patches_per_chunk, num_extra_chunks,
                 augmentation_params, 
                 dataset_name,
                 skip_last_event=False, sample_repetitions=1, preproc_polarity=None, 
                 custom_sampler = True,
                 workers=8, pin_memory=False, classes_to_exclude=[], balance=None):
        super().__init__()
        self.batch_size = batch_size
        self.chunk_len_ms = chunk_len_ms
        self.patch_size = patch_size
        self.min_activations_per_patch = min_activations_per_patch
        self.bins = bins
        self.min_patches_per_chunk = min_patches_per_chunk
        self.num_extra_chunks = num_extra_chunks
        
        self.augmentation_params = augmentation_params
        self.dataset_name = dataset_name
        self.workers = workers
        self.sample_repetitions = sample_repetitions
        self.preproc_polarity = preproc_polarity
        self.skip_last_event = skip_last_event
        self.pin_memory = pin_memory
        self.classes_to_exclude = classes_to_exclude
        
        self.pre_padding = True
        self.custom_sampler = custom_sampler
        
        
        self.dataset_name = dataset_name
        if dataset_name == 'DVS128':
            self.data_folder = './datasets/DvsGesture/clean_dataset_frames_12000/'
            self.width, self.height = 128, 128
            self.num_classes = 12 - len(classes_to_exclude)
            self.class_mapping = copy.deepcopy(DVS128_class_mapping)
            for c in classes_to_exclude: del self.class_mapping[c]
            self.class_mapping = { i:l[1] for i,l in enumerate(sorted(self.class_mapping.items(), key=lambda x:x[0])) }
        elif dataset_name == 'ASL_DVS':
            self.data_folder = './datasets/ICCV2019_DVS_dataset/clean_dataset_frames_2000/'
            self.width, self.height = 240, 180
            self.num_classes = 24
            self.class_mapping = { i:l for i,l in enumerate('a b c d e f g h i k l m n o p q r s t u v w x y'.split()) }
        elif dataset_name == 'SLAnimals_3s':
            self.data_folder = './datasets/SL_animal_splits/dataset_3sets_12000/'
            self.width, self.height = 128, 128
            self.num_classes = 19
            self.class_mapping = { i:l for i,l in enumerate(range(self.num_classes)) }
        elif dataset_name == 'SLAnimals_4s':
            self.data_folder = './datasets/SL_animal_splits/dataset_4sets_12000/'
            self.width, self.height = 128, 128
            self.num_classes = 19
            self.class_mapping = { i:l for i,l in enumerate(range(self.num_classes)) }
        elif dataset_name == 'Caltech':
            self.data_folder = './datasets/N_Caltech_101/clean_dataset_frames_2000/'
            self.width, self.height = 240, 180
            self.num_classes = 101
            self.class_mapping = { i:l for i,l in enumerate(range(self.num_classes)) }
        else: raise ValueError(f'Dataset [{dataset_name}] not handled')
        

        self.preproc_event_size = 1 if '1' in self.preproc_polarity else 2
        self.preproc_event_size *= bins
        self.token_dim = patch_size*patch_size * self.preproc_event_size


    def custom_collate_fn(self, batch_samples):
        pols, pixels, labels = [], [], []
        
        for num_sample, sample in enumerate(batch_samples): 
            # Sample -> time_sequence
            # #samples == batch_size
            
            if sample is None or len(sample[0]) == 0: 
                print('Empty sample')
                print(len(sample), len(sample[0]))
                continue
            
            pols.append(sample[0])
            pixels.append(sample[1])
            labels.append(sample[2])
        
        if len(pols) == 0: return None, None, None
        token_size = pols[0][0].shape[-1]
            
        pols = pad_list_of_sequences(pols, token_size, self.pre_padding)
        pixels = pad_list_of_sequences(pixels, 2, self.pre_padding)
        
        pols, pixels, labels = pols, pixels.long(), torch.tensor(labels).long()
        return pols, pixels, labels
    
    
    def train_dataloader(self):
        dt = EventDataset(self.data_folder+'train/', chunk_len_ms = self.chunk_len_ms, 
                           validation=False, 
                           preproc_polarity=self.preproc_polarity, patch_size=self.patch_size,
                           min_activations_per_patch=self.min_activations_per_patch,
                           bins = self.bins, 
                           min_patches_per_chunk = self.min_patches_per_chunk,
                           num_extra_chunks = self.num_extra_chunks,
                           dataset_name=self.dataset_name, height=self.height, width=self.width,
                           augmentation_params=self.augmentation_params, 
                           classes_to_exclude=self.classes_to_exclude)
        if self.custom_sampler: 
            sampler = CustomBatchSampler(batch_size=self.batch_size, label_dict=dt.get_label_dict(), sample_repetitions=self.sample_repetitions)
            dl = DataLoader(dt, batch_sampler=sampler, collate_fn=self.custom_collate_fn, num_workers=self.workers, pin_memory=self.pin_memory)
        else:
            dl = DataLoader(dt, batch_size=self.batch_size, collate_fn=self.custom_collate_fn, shuffle=True, num_workers=self.workers, pin_memory=self.pin_memory)
        return dl
    def val_dataloader(self):
        dt = EventDataset(self.data_folder+'test/', chunk_len_ms = self.chunk_len_ms, 
                           validation=True, 
                           preproc_polarity=self.preproc_polarity, patch_size=self.patch_size,
                           min_activations_per_patch=self.min_activations_per_patch,
                           bins = self.bins, 
                           min_patches_per_chunk = self.min_patches_per_chunk,
                           num_extra_chunks = self.num_extra_chunks,
                           dataset_name=self.dataset_name, height=self.height, width=self.width,
                           augmentation_params=self.augmentation_params, 
                           classes_to_exclude=self.classes_to_exclude)
        dl = DataLoader(dt, batch_size=(self.batch_size//2)+1, shuffle=False, collate_fn=self.custom_collate_fn, num_workers=self.workers, pin_memory=self.pin_memory)
        return dl
    
    



