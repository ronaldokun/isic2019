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
from base import BASE_PATH


params = {}

## Base Path
params[path_base] = BASE_PATH
## Save summaries and model here
params['save_dir'] = params['path_base']/'out'
## Data is loaded from here
params['data_dir'] = params_['path_base'] / 'data'

# Model Selection
params['model_type'] = 'efficientnet-b0'
params['dataset_names'] = ['full'] #['official']#,'sevenpoint_rez3_ll']
params['file_ending'] = '.jpg'
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

## Deterministic cropping
params['deterministic_eval'] = True
params['numCropPositions'] = 1
num_scales = 4
all_scales = [1.0,0.5,0.75,0.25,0.9,0.6,0.4]
params['cropScales'] = all_scales[:num_scales]
params['cropFlipping'] = 4
params['multiCropEval'] = params['numCropPositions']*len(params['cropScales'])*params['cropFlipping']
params['offset_crop'] = 0.2    

## Scale up for b1-b7
params['input_size'] = [224,224,3]     

### Training Parameters ###
# Batch size
params['batchSize'] = 20#*len(params['numGPUs'])
# Initial learning rate
params['learning_rate'] = 0.00015#*len(params['numGPUs'])
# Lower learning rate after no improvement over 100 epochs
params['lowerLRAfter'] = 25
# If there is no validation set, start lowering the LR after X steps
params['lowerLRat'] = 50
# Divide learning rate by this value
params['LRstep'] = 5
# Maximum number of training iterations
params['training_steps'] = 20 #250
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

# Data AUG
#params['full_color_distort'] = True
params['autoaugment'] = False     
params['flip_lr_ud'] = True
params['full_rot'] = 180
params['scale'] = (0.8,1.2)
params['shear'] = 10
params['cutout'] = 16

# Meta settings
params['meta_features'] = ['age_num','sex_oh','loc_oh']
params['meta_feature_sizes'] = [1,8,2]
params['encode_nan'] = False
params['model_load_path'] = '/out/2020.test_effb0_rr'
params['fc_layers_before'] = [256,256]
params['fc_layers_after'] = [1024]
params['freeze_cnn'] = True
params['learning_rate_meta'] = 0.00001
# each feature is set to missing with this prob
params['drop_augment'] = 0.1
params['dropout_meta'] = 0.4
params['scale_features'] = True

### Data ###
params['preload'] = False
# Labels first
# Targets, as dictionary, indexed by im file name
params['labels_dict'] = {}
path1 = params['data_dir'] + '/labels/'
 # All sets
allSets = glob(path1 + '*/')   
# Go through all sets
for i in range(len(allSets)):
    # Check if want to include this dataset
    foundSet = False
    for j in range(len(params['dataset_names'])):
        if params['dataset_names'][j] in allSets[i]:
            foundSet = True
    if not foundSet:
        continue                
    # Find csv file
    files = sorted(glob(allSets[i]+'*'))
    for j in range(len(files)):
        if 'csv' in files[j]:
            break
    # Load csv file
    with open(files[j], newline='') as csvfile:
        labels_str = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in labels_str:
            if 'image_id' == row[0]:
                continue
            #if 'ISIC' in row[0] and '_downsampled' in row[0]:
            #    print(row[0])
            if row[0] + '_downsampled' in params['labels_dict']:
                print("removed",row[0] + '_downsampled')
                continue
            if params['numClasses'] == 2:
                params['labels_dict'][row[0]] = np.array([int(float(row[1])),int(float(row[2]))])                    
            if params['numClasses'] == 7:
                params['labels_dict'][row[0]] = np.array([int(float(row[1])),int(float(row[2])),int(float(row[3])),int(float(row[4])),int(float(row[5])),int(float(row[6])),int(float(row[7]))])
            elif params['numClasses'] == 8:
                if len(row) < 9 or row[8] == '':
                    class_8 = 0
                else:
                    class_8 = int(float(row[8]))
                params['labels_dict'][row[0]] = np.array([int(float(row[1])),int(float(row[2])),int(float(row[3])),int(float(row[4])),int(float(row[5])),int(float(row[6])),int(float(row[7])),class_8])
            elif params['numClasses'] == 9:
                if len(row) < 9 or row[8] == '':
                    class_8 = 0
                else:
                    class_8 = int(float(row[8]))  
                if len(row) < 10 or row[9] == '':
                    class_9 = 0
                else:
                    class_9 = int(float(row[9]))                                           
                params['labels_dict'][row[0]] = np.array([int(float(row[1])),int(float(row[2])),int(float(row[3])),int(float(row[4])),int(float(row[5])),int(float(row[6])),int(float(row[7])),class_8,class_9])
# Save all im paths here
params['im_paths'] = []
params['labels_list'] = []
# Define the sets
path1 = params['data_dir'] + '/images/'
# All sets
allSets = sorted(glob(path1 + '*/'))
# Ids which name the folders
# Make official first datasets
for i in range(len(allSets)):
    if params['dataset_names'][0] in allSets[i]:
        temp = allSets[i]
        allSets.remove(allSets[i])
        allSets.insert(0, temp)
print(allSets)        
# Set of keys, for marking old HAM10000
params['key_list'] = []
if params['exclude_inds']:
    with open(params['save_dir'] + 'indices_exclude.pkl','rb') as f:
        indices_exclude = pickle.load(f)          
    exclude_list = []    
for i in range(len(allSets)):
    # All files in that set
    files = sorted(glob(allSets[i]+'*'))
    # Check if there is something in there, if not, discard
    if len(files) == 0:
        continue
    # Check if want to include this dataset
    foundSet = False
    for j in range(len(params['dataset_names'])):
        if params['dataset_names'][j] in allSets[i]:
            foundSet = True
    if not foundSet:
        continue                    
    for j in tqdm(range(len(files))):
        if '.jpg' in files[j] or '.jpeg' in files[j] or '.JPG' in files[j] or '.JPEG' in files[j] or '.png' in files[j] or '.PNG' in files[j]:                
            # Add according label, find it first
            found_already = False
            for key in params['labels_dict']:
                if key + params['file_ending'] in files[j]:
                    if found_already:
                        print("Found already:",key,files[j])                     
                    params['key_list'].append(key)
                    params['labels_list'].append(params['labels_dict'][key])
                    found_already = True
            if found_already:
                params['im_paths'].append(files[j])     
                if params['exclude_inds']:
                    for key in indices_exclude:
                        if key in files[j]:
                            exclude_list.append(indices_exclude[key])                                       
# Convert label list to array
params['labels_array'] = np.array(params['labels_list'])
print(np.mean(params['labels_array'],axis=0))        
# Create indices list with HAM10000 only
params['HAM10000_inds'] = []
HAM_START = 24306
HAM_END = 34320
for j in range(len(params['key_list'])):
    try:
        curr_id = [int(s) for s in re.findall(r'\d+',params['key_list'][j])][-1]
    except:
        continue
    if curr_id >= HAM_START and curr_id <= HAM_END:
            params['HAM10000_inds'].append(j)
    params['HAM10000_inds'] = np.array(params['HAM10000_inds'])    
    print("Len ham",len(params['HAM10000_inds']))   
    # Perhaps preload images
    if params['preload']:
        params['images_array'] = np.zeros([len(params['im_paths']),params['input_size_load'][0],params['input_size_load'][1],params['input_size_load'][2]],dtype=np.uint8)
        for i in range(len(params['im_paths'])):
            x = scipy.ndimage.imread(params['im_paths'][i])
            #x = x.astype(np.float32)   
            # Scale to 0-1 
            #min_x = np.min(x)
            #max_x = np.max(x)
            #x = (x-min_x)/(max_x-min_x)
            params['images_array'][i,:,:,:] = x
            if i%1000 == 0:
                print(i+1,"images loaded...")     
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

    ### Define Indices ###
    # Just divide into 5 equally large sets
    with open(params['save_dir'] + 'indices_isic2020.pkl','rb') as f:
        indices = pickle.load(f)           
    params['trainIndCV'] = indices['trainIndCV']
    params['valIndCV'] = indices['valIndCV']
    if params['exclude_inds']:
        exclude_list = np.array(exclude_list)
        all_inds = np.arange(len(params['im_paths']))
        exclude_inds = all_inds[exclude_list.astype(bool)]
        for i in range(len(params['trainIndCV'])):
            params['trainIndCV'][i] = np.setdiff1d(params['trainIndCV'][i],exclude_inds)
        for i in range(len(params['valIndCV'])):
            params['valIndCV'][i] = np.setdiff1d(params['valIndCV'][i],exclude_inds)     
    # Consider case with more than one set
    if len(params['dataset_names']) > 1:
        restInds = np.array(np.arange(25331,params['labels_array'].shape[0]))
        for i in range(params['numCV']):
            params['trainIndCV'][i] = np.concatenate((params['trainIndCV'][i],restInds))        
    print("Train")
    for i in range(len(params['trainIndCV'])):
        print(params['trainIndCV'][i].shape)
    print("Val")
    for i in range(len(params['valIndCV'])):
        print(params['valIndCV'][i].shape)    

    # Use this for ordered multi crops
    if params['orderedCrop']:
        # Crop positions, always choose multiCropEval to be 4, 9, 16, 25, etc.
        params['cropPositions'] = np.zeros([len(params['im_paths']),params['multiCropEval'],2],dtype=np.int64)
        #params['imSizes'] = np.zeros([len(params['im_paths']),params['multiCropEval'],2],dtype=np.int64)
        for u in range(len(params['im_paths'])):
            height, width = imagesize.get(params['im_paths'][u])
            if width < params['input_size'][0]:
                height = int(params['input_size'][0]/float(width))*height
                width = params['input_size'][0]
            if height < params['input_size'][0]:
                width = int(params['input_size'][0]/float(height))*width
                height = params['input_size'][0]            
            ind = 0
            for i in range(np.int32(np.sqrt(params['multiCropEval']))):
                for j in range(np.int32(np.sqrt(params['multiCropEval']))):
                    params['cropPositions'][u,ind,0] = params['input_size'][0]/2+i*((width-params['input_size'][1])/(np.sqrt(params['multiCropEval'])-1))
                    params['cropPositions'][u,ind,1] = params['input_size'][1]/2+j*((height-params['input_size'][0])/(np.sqrt(params['multiCropEval'])-1))
                    #params['imSizes'][u,ind,0] = curr_im_size[0]

                    ind += 1
        # Sanity checks
        #print("Positions",params['cropPositions'])
        # Test image sizes
        height = params['input_size'][0]
        width = params['input_size'][1]
        for u in range(len(params['im_paths'])):
            height_test, width_test = imagesize.get(params['im_paths'][u])
            if width_test < params['input_size'][0]:
                height_test = int(params['input_size'][0]/float(width_test))*height_test
                width_test = params['input_size'][0]
            if height_test < params['input_size'][0]:
                width_test = int(params['input_size'][0]/float(height_test))*width_test
                height_test = params['input_size'][0]                
            test_im = np.zeros([width_test,height_test]) 
            for i in range(params['multiCropEval']):
                im_crop = test_im[np.int32(params['cropPositions'][u,i,0]-height/2):np.int32(params['cropPositions'][u,i,0]-height/2)+height,np.int32(params['cropPositions'][u,i,1]-width/2):np.int32(params['cropPositions'][u,i,1]-width/2)+width]
                if im_crop.shape[0] != params['input_size'][0]:
                    print("Wrong shape",im_crop.shape[0],params['im_paths'][u])    
                if im_crop.shape[1] != params['input_size'][1]:
                    print("Wrong shape",im_crop.shape[1],params['im_paths'][u])      
                    
    pd.to_pickle(params, 'params.pkl')
    
    return params
