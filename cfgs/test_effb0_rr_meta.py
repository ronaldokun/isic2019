import os
import sys
import h5py
import re
import csv
import numpy as np
from glob import glob
import scipy
import pickle
import imagesize
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from .data import BASE_PATH, ordered_crop

params = {}

# ## Paths

## Base Path
params['path_base'] = BASE_PATH
## Save summaries and model here
params['save_dir'] = params['path_base'] / 'out'
## Data is loaded from here
params['data_dir'] = params['path_base'] / 'data'
## CV Indices
params['indices'] = params['save_dir'] / 'indices_isic2020.pkl'

# ## Model Selection

params['model_type'] = 'efficientnet-b0'
params['dataset_names'] = ['full']
params['file_ending'] = '.jpg'
params['input_size_load'] = [512, 512, 3]
params['exclude_inds'] = False
params['same_sized_crops'] = False
params['multiCropEval'] = 9
params['var_im_size'] = False
params['orderedCrop'] = False
params['voting_scheme'] = 'average'    
params['classification'] = True
params['balance_classes'] = 2
params['extra_fac'] = 1.0
params['numClasses'] = 2
params['no_c9_eval'] = True
params['numOut'] = params['numClasses']
params['numCV'] = 5
params['trans_norm_first'] = True

# ## Deterministic cropping

params['deterministic_eval'] = True
params['numCropPositions'] = 1
num_scales = 4
all_scales = [1.0,0.5,0.75,0.25,0.9,0.6,0.4]
params['cropScales'] = all_scales[:num_scales]
params['cropFlipping'] = 4
params['multiCropEval'] =  params['numCropPositions']*len(params['cropScales'])*params['cropFlipping']
params['numCropPositions']*len(params['cropScales'])*params['cropFlipping']
params['offset_crop'] = 0.2    

# ## Scale up for b1-b7

params['input_size'] = [224,224,3]     

# + [markdown]
# # Training Parameters
# -

# Batch size
params['batchSize'] = 20#*len(params['numGPUs'])
# Initial learning rate
params['learning_rate'] = 0.001#*len(params['numGPUs'])
# Lower learning rate after no improvement over 100 epochs
params['lowerLRAfter'] = 25
# If there is no validation set, start lowering the LR after X steps
params['lowerLRat'] = 50
# Divide learning rate by this value
params['LRstep'] = 5
# Maximum number of training iterations
params['training_steps'] = 40 #250
# Display error every X steps
params['display_step'] = 1
# Scale?
params['scale_targets'] = False
# Peak at test error during training? (generally, dont do this!)
params['peak_at_testerr'] = False
# Print trainerr
params['print_trainerr'] = True
# Subtract trainset mean?
params['subtract_set_mean'] = False
params['setMean'] = np.array([0.0, 0.0, 0.0])   
params['setStd'] = np.array([1.0, 1.0, 1.0])   

# # Data AUG

#params['full_color_distort'] = True
params['autoaugment'] = False     
params['flip_lr_ud'] = True
params['full_rot'] = 180
params['scale'] = (0.8,1.2)
params['shear'] = 10
params['cutout'] = 16

# # Meta settings

params['meta_features'] = ['age_num','sex_oh','loc_oh']
params['meta_feature_sizes'] = [1,8,2]
params['encode_nan'] = False
params['model_load_path'] = '/out/test_effb0_rr'
params['fc_layers_before'] = [256,256]
params['fc_layers_after'] = [1024]
params['freeze_cnn'] = True
params['learning_rate_meta'] = 0.00001
# each feature is set to missing with this prob
params['drop_augment'] = 0.1
params['dropout_meta'] = 0.4
params['scale_features'] = True

# # Data
# Check labels first 
# img_name are the keys
# one-hot encoding targets as arrays are the values

params['preload'] = False
params['labels_dict'] = {}
all_sets = params['data_dir'] / 'labels'
params['img_paths'] = []
params['labels_list'] = []
params['key_list'] = []
img_dirs = params['data_dir'] / 'images'

# Go through all sets
for p in all_sets.iterdir():
    if p.is_dir() and p.name in params['dataset_names']:
        for file in p.iterdir():
            if file.suffix == '.csv':
                df = pd.read_csv(file)
                keys = df.image_id.values
                targets = df.drop('image_id', axis=1).values
                params['labels_dict'].update(dict(zip(keys, targets)))

# ## Images

for img_dir in img_dirs.iterdir():
    if img_dir.is_dir():
        params['img_paths'].extend([img for img in img_dir.iterdir() if img.suffix.lower() in ('.jpg', '.jpeg', '.png')])

for img in params['img_paths']:
    if img.stem in params['labels_dict']:
        params['key_list'].append(img.stem)
        params['labels_list'].append(params['labels_dict'][img.stem])    

# Convert label list to array
params['labels_array'] = np.array(params['labels_list'])

# +
# Perhaps preload images
if params['preload']:
    params['images_array'] = np.zeros([len(params['im_paths']),params['input_size_load'][0],params['input_size_load'][1],params['input_size_load'][2]],dtype=np.uint8)
    for i in range(len(params['img_paths'])):
        x = scipy.ndimage.imread(params['img_paths'][i])
        #x = x.astype(np.float32)   
        # Scale to 0-1 
        #min_x = np.min(x)
        #max_x = np.max(x)
        #x = (x-min_x)/(max_x-min_x)
        params['images_array'][i,:,:,:] = x
        if i%1000 == 0:
            print(i+1,"images loaded...")   
        break

if params['subtract_set_mean']:
    params['images_means'] = np.zeros([len(params['im_paths']),3])
    for i in range(len(params['im_paths'])):
        x = scipy.ndimage.imread(params['im_paths'][i])
        x = x.astype(np.float32)   
        # Scale to 0-1 
        min_x = np.min(x)
        max_x = np.max(x)
        x = (x-min_x)/(max_x-min_x)
        params['images_means'][i,:] = np.mean(x,(0,1))
        if i%1000 == 0:
            print(i+1,"images processed for mean...")         
# -

### Define Indices ###
# Just divide into 5 equally large sets
with open(str(params['indices']), 'rb') as f:
    indices = pickle.load(f)           
params['trainIndCV'] = indices['trainIndCV']
params['valIndCV'] = indices['valIndCV']
if params['exclude_inds']:
    exclude_list = np.array(exclude_list)
    all_inds = np.arange(len(params['img_paths']))
    exclude_inds = all_inds[exclude_list.astype(bool)]
    for i in range(len(params['trainIndCV'])):
        params['trainIndCV'][i] = np.setdiff1d(params['trainIndCV'][i],exclude_inds)
    for i in range(len(params['valIndCV'])):
        params['valIndCV'][i] = np.setdiff1d(params['valIndCV'][i],exclude_inds) 

print("Train")
for i in range(len(params['trainIndCV'])):
    print(params['trainIndCV'][i].shape)
print("Val")
for i in range(len(params['valIndCV'])):
    print(params['valIndCV'][i].shape)    

# Use this for ordered multi crops
if params['orderedCrop']:
    params = ordered_crop(params)    
pd.to_pickle(params, 'params_rr_meta.pkl')


