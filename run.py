import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models as tv_models
from torch.utils.data import DataLoader
from torchsummary import summary
from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score, classification_report
import numpy as np
from scipy import io
import threading
import pickle
from pathlib import Path
import math
import os
import random
import sys
from glob import glob
import re
import gc
import importlib
import time
import sklearn.preprocessing
import utils
from sklearn.utils import class_weight
import psutil
import models
from tqdm import tqdm
# from fastprogress import progress_bar as tqdm
# from fastprogress import master_bar as mb
from cfgs.test_effb0_ss import params
# %reload_ext autoreload
# %autoreload 2

# +
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(42)


# -

def getErrClassification_mgpu(mdlParams, indices, modelVars, exclude_class=None):
    """Helper function to return the error of a set
    Args:
      mdlParams: dictionary, configuration file
      indices: string, either "trainInd", "valInd" or "testInd"
    Returns:
      loss: float, avg loss
      acc: float, accuracy
      sensitivity: float, sensitivity
      spec: float, specificity
      conf: float matrix, confusion matrix
    """
    # Set up sizes
    if indices == 'trainInd':
        numBatches = int(math.floor(len(mdlParams[indices])/mdlParams['batchSize']/mdlParams['numGPUs']))
    else:
        numBatches = int(math.ceil(len(mdlParams[indices])/mdlParams['batchSize']/mdlParams['numGPUs']))
    # Consider multi-crop case
    if mdlParams.get('eval_flipping',0) > 1 and mdlParams.get('multiCropEval',0) > 0:
        loss_all = np.zeros([numBatches])
        predictions = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
        targets = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])        
        loss_mc = np.zeros([len(mdlParams[indices])*mdlParams['eval_flipping']])
        predictions_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval'],mdlParams['eval_flipping']])
        targets_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval'],mdlParams['eval_flipping']])  
        # Very suboptimal method
        ind = -1
        for i, (inputs, labels, inds, flip_ind) in enumerate(modelVars['dataloader_'+indices]):
            if flip_ind[0] != np.mean(np.array(flip_ind)):
                print("Problem with flipping",flip_ind)
            if flip_ind[0] == 0:
                ind += 1
            # Get data
            if mdlParams.get('meta_features',None) is not None: 
                inputs[0] = inputs[0].cuda()
                inputs[1] = inputs[1].cuda()
            else:            
                inputs = inputs.to(modelVars['device'])
            labels = labels.to(modelVars['device'])       
            # Not sure if thats necessary
            modelVars['optimizer'].zero_grad()    
            with torch.set_grad_enabled(False):
                # Get outputs
                if mdlParams.get('aux_classifier',False):
                    outputs, outputs_aux = modelVars['model'](inputs)
                    if mdlParams['eval_aux_classifier']:
                        outputs = outputs_aux
                else:
                    outputs = modelVars['model'](inputs)
                preds = modelVars['softmax'](outputs)      
                # Loss
                loss = modelVars['criterion'](outputs, labels)           
            # Write into proper arrays
            loss_mc[ind] = np.mean(loss.cpu().numpy())
            predictions_mc[ind,:,:,flip_ind[0]] = np.transpose(preds.cpu().numpy())
            tar_not_one_hot = labels.data.cpu().numpy()
            tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
            tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
            targets_mc[ind,:,:,flip_ind[0]] = np.transpose(tar)
        # Targets stay the same
        targets = targets_mc[:,:,0,0]
        # reshape preds
        predictions_mc = np.reshape(predictions_mc,[predictions_mc.shape[0],predictions_mc.shape[1],mdlParams['multiCropEval']*mdlParams['eval_flipping']])
        if mdlParams['voting_scheme'] == 'vote':
            # Vote for correct prediction
            print("Pred Shape",predictions_mc.shape)
            predictions_mc = np.argmax(predictions_mc,1)    
            print("Pred Shape",predictions_mc.shape) 
            for j in range(predictions_mc.shape[0]):
                predictions[j,:] = np.bincount(predictions_mc[j,:],minlength=mdlParams['numClasses'])   
            print("Pred Shape",predictions.shape) 
        elif mdlParams['voting_scheme'] == 'average':
            predictions = np.mean(predictions_mc,2)        
    elif mdlParams.get('multiCropEval',0) > 0:
        loss_all = np.zeros([numBatches])
        predictions = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
        targets = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])        
        loss_mc = np.zeros([len(mdlParams[indices])])
        predictions_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval']])
        targets_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval']])
        for i, (inputs, labels, inds) in tqdm(enumerate(modelVars['dataloader_'+indices]), total=len(mdlParams[indices])):
            # Get data
            if mdlParams.get('meta_features',None) is not None: 
                inputs[0] = inputs[0].cuda()
                inputs[1] = inputs[1].cuda()
            else:            
                inputs = inputs.to(modelVars['device'])
            labels = labels.to(modelVars['device'])       
            # Not sure if thats necessary
            modelVars['optimizer'].zero_grad()    
            with torch.set_grad_enabled(False):
                # Get outputs
                if mdlParams.get('aux_classifier',False):
                    outputs, outputs_aux = modelVars['model'](inputs)
                    if mdlParams['eval_aux_classifier']:
                        outputs = outputs_aux
                else:
                    outputs = modelVars['model'](inputs)
                preds = modelVars['softmax'](outputs)      
                # Loss
                loss = modelVars['criterion'](outputs, labels)           
            # Write into proper arrays
            loss_mc[i] = np.mean(loss.cpu().numpy())
            #print(f'predictions_mc shape: {predictions_mc[i].shape}')
            predictions_mc[i,:,:] = np.transpose(preds.cpu().numpy()) #[:,mdlParams['multiCropEval']-1]
            tar_not_one_hot = labels.data.cpu().numpy()
            tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
            tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
            targets_mc[i,:,:] = np.transpose(tar)
        # Targets stay the same
        targets = targets_mc[:,:,0]
        if mdlParams['voting_scheme'] == 'vote':
            # Vote for correct prediction
            print("Pred Shape",predictions_mc.shape)
            predictions_mc = np.argmax(predictions_mc,1)    
            print("Pred Shape",predictions_mc.shape) 
            for j in range(predictions_mc.shape[0]):
                predictions[j,:] = np.bincount(predictions_mc[j,:],minlength=mdlParams['numClasses'])   
            print("Pred Shape",predictions.shape) 
        elif mdlParams['voting_scheme'] == 'average':
            predictions = np.mean(predictions_mc,2)
    else:    
        if mdlParams.get('model_type_cnn') is not None and mdlParams['numRandValSeq'] > 0:
            loss_all = np.zeros([numBatches])
            predictions = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
            targets = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])        
            loss_mc = np.zeros([len(mdlParams[indices])])
            predictions_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['numRandValSeq']])
            targets_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['numRandValSeq']])   
            for i, (inputs, labels, inds) in enumerate(modelVars['dataloader_'+indices]):
                # Get data
                if mdlParams.get('meta_features',None) is not None: 
                    inputs[0] = inputs[0].cuda()
                    inputs[1] = inputs[1].cuda()
                else:            
                    inputs = inputs.to(modelVars['device'])
                labels = labels.to(modelVars['device'])       
                # Not sure if thats necessary
                modelVars['optimizer'].zero_grad()    
                with torch.set_grad_enabled(False):
                    # Get outputs
                    if mdlParams.get('aux_classifier',False):
                        outputs, outputs_aux = modelVars['model'](inputs)
                        if mdlParams['eval_aux_classifier']:
                            outputs = outputs_aux
                    else:
                        outputs = modelVars['model'](inputs)
                    preds = modelVars['softmax'](outputs)      
                    # Loss
                    loss = modelVars['criterion'](outputs, labels)           
                # Write into proper arrays
                loss_mc[i] = np.mean(loss.cpu().numpy())
                predictions_mc[i,:,:] = np.transpose(preds)
                tar_not_one_hot = labels.data.cpu().numpy()
                tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
                tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
                targets_mc[i,:,:] = np.transpose(tar)
            # Targets stay the same
            targets = targets_mc[:,:,0]
            if mdlParams['voting_scheme'] == 'vote':
                # Vote for correct prediction
                print("Pred Shape",predictions_mc.shape)
                predictions_mc = np.argmax(predictions_mc,1)    
                print("Pred Shape",predictions_mc.shape) 
                for j in range(predictions_mc.shape[0]):
                    predictions[j,:] = np.bincount(predictions_mc[j,:],minlength=mdlParams['numClasses'])   
                print("Pred Shape",predictions.shape) 
            elif mdlParams['voting_scheme'] == 'average':
                predictions = np.mean(predictions_mc,2)
        else:
            for i, (inputs, labels, indices) in enumerate(modelVars['dataloader_'+indices]):
                # Get data
                if mdlParams.get('meta_features',None) is not None: 
                    inputs[0] = inputs[0].cuda()
                    inputs[1] = inputs[1].cuda()
                else:            
                    inputs = inputs.to(modelVars['device'])
                labels = labels.to(modelVars['device'])       
                # Not sure if thats necessary
                modelVars['optimizer'].zero_grad()    
                with torch.set_grad_enabled(False):
                    # Get outputs
                    if mdlParams.get('aux_classifier',False):
                        outputs, outputs_aux = modelVars['model'](inputs)
                        if mdlParams['eval_aux_classifier']:
                            outputs = outputs_aux
                    else:
                        outputs = modelVars['model'](inputs)
                    #print("in",inputs.shape,"out",outputs.shape)
                    preds = modelVars['softmax'](outputs)      
                    # Loss
                    loss = modelVars['criterion'](outputs, labels)     
                # Write into proper arrays                
                if i==0:
                    loss_all = np.array([loss.cpu().numpy()])
                    predictions = preds.cpu().numpy()
                    tar_not_one_hot = labels.data.cpu().numpy()
                    tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
                    tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1   
                    targets = tar    
                    #print("Loss",loss_all)         
                else:                 
                    loss_all = np.concatenate((loss_all,np.array([loss.cpu().numpy()])),0)
                    predictions = np.concatenate((predictions,preds.cpu().numpy()),0)
                    tar_not_one_hot = labels.data.cpu().numpy()
                    tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
                    tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1                   
                    targets = np.concatenate((targets,tar),0)
                    #allInds[(i*len(mdlParams['numGPUs'])+k)*bSize:(i*len(mdlParams['numGPUs'])+k+1)*bSize] = res_tuple[3][k]
            predictions_mc = predictions
    #print("Check Inds",np.setdiff1d(allInds,mdlParams[indices]))
    # Calculate metrics
    if exclude_class is not None:
        predictions = np.concatenate((predictions[:,:exclude_class],predictions[:,exclude_class+1:]),1)
        targets = np.concatenate((targets[:,:exclude_class],targets[:,exclude_class+1:]),1)    
        num_classes = mdlParams['numClasses']-1
    elif mdlParams['numClasses'] == 9 and mdlParams.get('no_c9_eval',False):
        predictions = predictions[:,:mdlParams['numClasses']-1]
        targets = targets[:,:mdlParams['numClasses']-1]
        num_classes = mdlParams['numClasses']-1
    else:
        num_classes = mdlParams['numClasses']
    # Accuarcy
    acc = np.mean(np.equal(np.argmax(predictions,1),np.argmax(targets,1)))
    # Confusion matrix
    conf = confusion_matrix(np.argmax(targets,1),np.argmax(predictions,1))
    if conf.shape[0] < num_classes:
        conf = np.ones([num_classes,num_classes])
    # Class weighted accuracy
    wacc = conf.diagonal()/conf.sum(axis=1)    
    # Sensitivity / Specificity
    sensitivity = np.zeros([num_classes])
    specificity = np.zeros([num_classes])
    if num_classes > 2:
        for k in range(num_classes):
                sensitivity[k] = conf[k,k]/(np.sum(conf[k,:]))
                true_negative = np.delete(conf,[k],0)
                true_negative = np.delete(true_negative,[k],1)
                true_negative = np.sum(true_negative)
                false_positive = np.delete(conf,[k],0)
                false_positive = np.sum(false_positive[:,k])
                specificity[k] = true_negative/(true_negative+false_positive)
                # F1 score
                f1 = f1_score(np.argmax(predictions,1),np.argmax(targets,1),average='weighted')                
    else:
        tn, fp, fn, tp = confusion_matrix(np.argmax(targets,1),np.argmax(predictions,1)).ravel()
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        # F1 score
        f1 = f1_score(np.argmax(predictions,1),np.argmax(targets,1))
    # AUC
    fpr = {}
    tpr = {}
    roc_auc = np.zeros([num_classes])
    if num_classes > 9:
        print(predictions)
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(targets[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return np.mean(loss_all), acc, sensitivity, specificity, conf, f1, roc_auc, wacc, predictions, targets, predictions_mc 



OUT = 'test_effb0_ss'

# Path name from filename
params['save_dir_base'] = params['save_dir'] / OUT

params['save_dir_base'].mkdir(exist_ok=True)

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

# Check if there is a validation set, if not, evaluate train error instead
if 'valIndCV' in params or 'valInd' in params:
    eval_set = 'valInd'
    print("Evaluating on validation set during training.")
else:
    eval_set = 'trainInd'
    print("No validation set, evaluating on training set during training.")

# Check if there were previous ones that have alreary bin learned
prevFile = Path(params['save_dir_base'] / 'CV.pkl')
#print(prevFile)
if prevFile.exists():
    print("Part of CV already done")
    with open(params['save_dir_base'] / 'CV.pkl', 'rb') as f:
        allData = pickle.load(f)
else:
    allData = {}
    allData['f1Best'] = {}
    allData['sensBest'] = {}
    allData['specBest'] = {}
    allData['accBest'] = {}
    allData['waccBest'] = {}
    allData['aucBest'] = {}
    allData['convergeTime'] = {}
    allData['bestPred'] = {}
    allData['targets'] = {}


def check_cv(cv):
    # Check if this fold was already trained
    already_trained, load_old = False, False
    # Def current CV set
    params['trainInd'] = params['trainIndCV'][cv]
    params['valInd'] = params['valIndCV'][cv]
    # Def current path for saving stuff
    if 'valIndCV' in params:
        params['save_dir'] = params['save_dir_base'] / f'CVSet{cv}'
        if params['save_dir_base'].is_dir():
            if params['save_dir'].is_dir():
                all_max_iter = []
                for name in params['save_dir'].iterdir():
                    load_old = True
                    int_list = [int(s) for s in re.findall(r'\d+',str(name))]
                    if len(int_list) > 0:
                        all_max_iter.append(int_list[-1])
                all_max_iter = np.array(all_max_iter)
                if len(all_max_iter) > 0 and np.max(all_max_iter) >= params['training_steps']:
                    print("Fold %d already fully trained with %d iterations"%(cv,np.max(all_max_iter)))
                    already_trained = True
    else:
        params['save_dir'] = params['save_dir_base']
        
    return already_trained, load_old


def balance_classes():
    # balance classes
    if params['balance_classes'] in (1,2, 7, 11):
        class_weights = class_weight.compute_class_weight('balanced',np.unique(np.argmax(params['labels_array'][params['trainInd'],:],1)),np.argmax(params['labels_array'][params['trainInd'],:],1)) 
        class_weights = class_weights*params['extra_fac']
    elif params['balance_classes'] in (3,4):
        # Split training set by classes
        not_one_hot = np.argmax(params['labels_array'],1)
        params['class_indices'] = []
        for i in range(params['numClasses']):
            params['class_indices'].append(np.where(not_one_hot==i)[0])
            # Kick out non-trainind indices
            params['class_indices'][i] = np.setdiff1d(params['class_indices'][i],params['valInd'])
    elif params['balance_classes'] in (5,6,13):
        # Other class balancing loss
        class_weights = 1.0/np.mean(params['labels_array'][params['trainInd'],:],axis=0)        
    elif params['balance_classes'] == 9:
        # Only use official indicies for calculation
        print("Balance 9")
        indices_ham = params['trainInd'][params['trainInd'] < 25331]
        if params['numClasses'] == 9:
            class_weights_ = 1.0/np.mean(params['labels_array'][indices_ham,:8],axis=0)
            class_weights = np.zeros([params['numClasses']])
            class_weights[:8] = class_weights_
            class_weights[-1] = np.max(class_weights_)
        else:
            class_weights = 1.0/np.mean(params['labels_array'][indices_ham,:],axis=0)
            
    if isinstance(params['extra_fac'], float):
        class_weights = np.power(class_weights,params['extra_fac'])
    else:
        class_weights = class_weights*params['extra_fac']
        
    params['class_weights'] = class_weights

    print("Current class weights with extra",class_weights)


def get_loaders():
    # Set up dataloaders
    num_workers = psutil.cpu_count(logical=False)
    # For train
    dataset_train = utils.ISICDataset(params, 'trainInd')
    # For val
    dataset_val = utils.ISICDataset(params, 'valInd')
    params['len_train'] = len(dataset_train)
    params['len_val'] = len(dataset_val)
    if params['multiCropEval'] > 0:
        modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=params['multiCropEval'], shuffle=False, num_workers=num_workers, pin_memory=True)  
    else:
        modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=params['batchSize'], shuffle=False, num_workers=num_workers, pin_memory=True)               

    if params['balance_classes'] == 12 or params['balance_classes'] == 13:
        strat_sampler = utils.StratifiedSampler(params)
        modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=params['batchSize'], sampler=strat_sampler, num_workers=num_workers, pin_memory=True) 
    else:
        modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=params['batchSize'], shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True) 


def initialize_model(cv):
    # Define model 
    modelVars['model'] = models.getModel(params)() 
    # Load trained model
    if params.get('meta_features',None) is not None:
        # Find best checkpoint
        files = (params['model_load_path'] / f'CVSet{cv}').glob('/*')
        global_steps = np.zeros([len(files)])
        for i in range(len(files)):
            # Use meta files to find the highest index
            if 'best' not in files[i]:
                continue
            if 'checkpoint' not in files[i]:
                continue                
            # Extract global step
            nums = [int(s) for s in re.findall(r'\d+',str(files[i]))]
            global_steps[i] = nums[-1]
        # Create path with maximum global step found
        chkPath = params['model_load_path'] / f'CVSet{cv}' / 'checkpoint_best-{int(np.max(global_steps))}.pt'
        print("Restoring lesion-trained CNN for meta data training: ",chkPath)
        # Load
        state = torch.load(chkPath)
        # Initialize model
        curr_model_dict = modelVars['model'].state_dict()
        for name, param in state['state_dict'].items():
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if curr_model_dict[name].shape == param.shape:
                curr_model_dict[name].copy_(param)
            else:
                print("not restored",name,param.shape)  

    if 'Dense' in params['model_type']:
        if params['input_size'][0] != 224:
            modelVars['model'] = utils.modify_densenet_avg_pool(modelVars['model'])
        num_ftrs = modelVars['model'].classifier.in_features
        modelVars['model'].classifier = nn.Linear(num_ftrs, params['numClasses'])
        #print(modelVars['model'])
    elif 'dpn' in params['model_type']:
        num_ftrs = modelVars['model'].classifier.in_channels
        modelVars['model'].classifier = nn.Conv2d(num_ftrs,params['numClasses'],[1,1])
    elif 'efficient' in params['model_type']:
        # Do nothing, output is prepared
        num_ftrs = modelVars['model']._fc.in_features
        modelVars['model']._fc = nn.Linear(num_ftrs, params['numClasses'])    
    elif 'wsl' in params['model_type']:
        num_ftrs = modelVars['model'].fc.in_features
        modelVars['model'].fc = nn.Linear(num_ftrs, params['numClasses'])          
    else:
        num_ftrs = modelVars['model'].last_linear.in_features
        modelVars['model'].last_linear = nn.Linear(num_ftrs, params['numClasses'])    
    # Take care of meta case
    if params.get('meta_features',None) is not None:
        # freeze cnn first
        if params['freeze_cnn']:
            # deactivate all
            for param in modelVars['model'].parameters():
                param.requires_grad = False            
            if 'efficient' in params['model_type']:
                # Activate fc
                for param in modelVars['model']._fc.parameters():
                    param.requires_grad = True
            elif 'wsl' in params['model_type']:
                # Activate fc
                for param in modelVars['model'].fc.parameters():
                    param.requires_grad = True
            else:
                # Activate fc
                for param in modelVars['model'].last_linear.parameters():
                    param.requires_grad = True                                
        else:
            # mark cnn parameters
            for param in modelVars['model'].parameters():
                param.is_cnn_param = True
            # unmark fc
            for param in modelVars['model']._fc.parameters():
                param.is_cnn_param = False                              
        # modify model
        modelVars['model'] = models.modify_meta(params,modelVars['model'])  
        # Mark new parameters
        for param in modelVars['model'].parameters():
            if not hasattr(param, 'is_cnn_param'):
                param.is_cnn_param = False                 
    # multi gpu support
    if params['numGPUs'] > 1:
        modelVars['model'] = nn.DataParallel(modelVars['model']) 
    modelVars['model'] = modelVars['model'].cuda()    


def define_loss():
    class_weights = params['class_weights']
    # Loss, with class weighting
    if params.get('focal_loss',False):
        modelVars['criterion'] = utils.FocalLoss(alpha=class_weights.tolist())
    elif params['balance_classes'] == 2:
        #modelVars['criterion'] = nn.BCEWithLogitsLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)))
        modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)))
    elif params['balance_classes'] == 3 or params['balance_classes'] == 0 or params['balance_classes'] == 12:
        modelVars['criterion'] = nn.CrossEntropyLoss()
    elif params['balance_classes'] == 8:
        modelVars['criterion'] = nn.CrossEntropyLoss(reduce=False)
    elif params['balance_classes'] == 6 or params['balance_classes'] == 7:
        modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)),reduce=False)
    elif params['balance_classes'] == 10:
        modelVars['criterion'] = utils.FocalLoss(params['numClasses'])
    elif params['balance_classes'] == 11:
        modelVars['criterion'] = utils.FocalLoss(params['numClasses'],alpha=torch.cuda.FloatTensor(class_weights.astype(np.float32)))
    else:
        modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)))    


# +
def define_optimizer():
    if params.get('meta_features',None) is not None:
        if params['freeze_cnn']:
            modelVars['optimizer'] = optim.AdamW(filter(lambda p: p.requires_grad, modelVars['model'].parameters()), lr=params['learning_rate_meta'])
            # sanity check
            for param in filter(lambda p: p.requires_grad, modelVars['model'].parameters()):
                print(param.name,param.shape)
        else:
            modelVars['optimizer'] = optim.AdamW([
                                                {'params': filter(lambda p: not p.is_cnn_param, modelVars['model'].parameters()), 'lr': params['learning_rate_meta']},
                                                {'params': filter(lambda p: p.is_cnn_param, modelVars['model'].parameters()), 'lr': params['learning_rate']}
                                                ], lr=params['learning_rate'])
    else:
        modelVars['optimizer'] = optim.AdamW(modelVars['model'].parameters(), lr=params['learning_rate'])

    # Decay LR by a factor of 0.1 every 7 epochs
#     modelVars['scheduler'] = lr_scheduler.StepLR(modelVars['optimizer'], step_size=params['lowerLRAfter'], gamma=1/np.float32(params['LRstep']))
    
#     modelVars['scheduler'] = lr_scheduler.OneCycleLR(modelVars['optimizer'], 
#                                                      max_lr=params['learning_rate'],
#                                                      epochs=params['training_steps'],
#                                                      steps_per_epoch=params['len_train']//params['batchSize'])
    
    modelVars['scheduler'] = lr_scheduler.ReduceLROnPlateau(modelVars['optimizer'], mode='max',
                                                           factor=0.1, patience=3, verbose=True, 
                                                           threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)     



# -

def restore_checkpoint(load_old):
    if load_old:
        # Find last, not last best checkpoint
        files = list(params['save_dir'].iterdir())
        global_steps = np.zeros([len(files)])
        for i, file in enumerate(files):
            # Use meta files to find the highest index
            if 'best' in str(file):
                continue
            if 'checkpoint-' not in str(file):
                continue                
            # Extract global step
            nums = [int(s) for s in re.findall(r'\d+',str(file))]
            global_steps[i] = nums[-1]
        # Create path with maximum global step found
        chkPath = params['save_dir'] / f'checkpoint-{int(np.max(global_steps))}.pt'
        print("Restoring: ",chkPath)
        # Load
        state = torch.load(chkPath)
        # Initialize model and optimizer
        modelVars['model'].load_state_dict(state['state_dict'])
        modelVars['optimizer'].load_state_dict(state['optimizer'])     
        start_epoch = state['epoch']+1
        params['valBest'] = state.get('valBest',1000)
        params['lastBestInd'] = state.get('lastBestInd',int(np.max(global_steps)))
    else:
        start_epoch = 1
        params['lastBestInd'] = -1
        # Track metrics for saving best model
        params['valBest'] = 1000

    return start_epoch


def train_fn():
    modelVars['model'].train()
    
    train_targets=[]
    train_outputs=[]
    
    # Num batches
    numBatchesTrain = int(math.floor(len(params['trainInd'])/params['batchSize']))   
    
    for j, (inputs, labels, indices) in tqdm(enumerate(modelVars['dataloader_trainInd']), total=numBatchesTrain):    
        # Run optimization        
        if params.get('meta_features',None) is not None: 
            inputs[0] = inputs[0].cuda()
            inputs[1] = inputs[1].cuda()
        else:
            inputs = inputs.cuda()
        labels = labels.cuda()        
        # zero the parameter gradients
        modelVars['optimizer'].zero_grad()             
        # forward
        with torch.set_grad_enabled(True):             
            if params.get('aux_classifier',False):
                outputs, outputs_aux = modelVars['model'](inputs) 
                loss1 = modelVars['criterion'](outputs, labels)
                labels_aux = labels.repeat(params['multiCropTrain'])
                loss2 = modelVars['criterion'](outputs_aux, labels_aux) 
                loss = loss1 + params['aux_classifier_loss_fac']*loss2     
            else:               
                outputs = modelVars['model'](inputs)     
                loss = modelVars['criterion'](outputs, labels)
            # backward + optimize only if in training phase
            loss.backward()
            modelVars['optimizer'].step()
        
        train_targets.extend(labels.cpu().detach().numpy().astype(int).tolist())
        train_outputs.extend(outputs)
            
        
    return loss.item(),train_outputs,train_targets


def eval_fn(cv, step):
    
    # Get metrics
    loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, _ = getErrClassification_mgpu(params, eval_set, modelVars)
    # Save in mat
    save_dict['loss'].append(loss)
    save_dict['acc'].append(accuracy)
    save_dict['wacc'].append(waccuracy)
    save_dict['auc'].append(auc)
    save_dict['sens'].append(sensitivity)
    save_dict['spec'].append(specificity)
    save_dict['f1'].append(f1)
    save_dict['step_num'].append(step)
    if os.path.isfile(params['save_dir'] / f'progression_{eval_set}.mat'):
        os.remove(params['save_dir'] / f'progression_{eval_set}.mat')                
    io.savemat(str(params['save_dir'] / f'progression_{eval_set}.mat'),save_dict)                
    eval_metric = -np.mean(waccuracy)
    # Check if we have a new best value
    if eval_metric < params['valBest']:
        params['valBest'] = eval_metric
        if params['classification']:
            allData['f1Best'][cv] = f1
            allData['sensBest'][cv] = sensitivity
            allData['specBest'][cv] = specificity
            allData['accBest'][cv] = accuracy
            allData['waccBest'][cv] = waccuracy
            allData['aucBest'][cv] = auc
        oldBestInd = params['lastBestInd']
        params['lastBestInd'] = step
        allData['convergeTime'][cv] = step
        # Save best predictions
        allData['bestPred'][cv] = predictions
        allData['targets'][cv] = targets
        # Write to File
        with open(params['save_dir_base'] / 'CV.pkl', 'wb') as f:
            pickle.dump(allData, f, pickle.HIGHEST_PROTOCOL)                 
        # Delte previously best model
        if (params['save_dir'] / f'checkpoint_best-{oldBestInd}.pt').is_file():
            (params['save_dir'] / f'checkpoint_best-{oldBestInd}.pt').unlink()
        # Save currently best model
        state = {'epoch': step, 'valBest': params['valBest'], 'lastBestInd': params['lastBestInd'], 'state_dict': modelVars['model'].state_dict(),'optimizer': modelVars['optimizer'].state_dict()}
        torch.save(state, params['save_dir'] / f'checkpoint_best-{step}.pt')
        
    # If its not better, just save it delete the last checkpoint if it is not current best one
    # Save current model
    state = {'epoch': step, 'valBest': params['valBest'], 'lastBestInd': params['lastBestInd'], 'state_dict': modelVars['model'].state_dict(),'optimizer': modelVars['optimizer'].state_dict()}
    torch.save(state, params['save_dir'] / f'checkpoint-{step}.pt')                           
    # Delete last one
    if step == params['display_step']:
        lastInd = 1
    else:
        lastInd = step-params['display_step']
    if (params['save_dir'] / f'checkpoint-{lastInd}.pt').is_file():
        (params['save_dir'] / f'checkpoint-{lastInd}.pt').unlink()
        
    return loss.item(),predictions,targets  


# Take care of CV
if params.get('cv_subset',None) is not None:
    cv_set = params['cv_subset']
else:
    cv_set = range(params['numCV'])
for cv in cv_set:      
    already_trained, load_old = check_cv(cv)    
    if already_trained:
        continue        
    print("CV set",cv)
    # Reset model graph 
    importlib.reload(models)
    # Collect model variables
    modelVars = {}
    #print("here")
    modelVars['device'] = device
    print(modelVars['device'])
    # Save training progress in here
    save_dict = {}
    save_dict['acc'] = []
    save_dict['loss'] = []
    save_dict['wacc'] = []
    save_dict['auc'] = []
    save_dict['sens'] = []
    save_dict['spec'] = []
    save_dict['f1'] = []
    save_dict['step_num'] = []
    if params['print_trainerr']:
        save_dict_train = {}
        save_dict_train['acc'] = []
        save_dict_train['loss'] = []
        save_dict_train['wacc'] = []
        save_dict_train['auc'] = []
        save_dict_train['sens'] = []
        save_dict_train['spec'] = []
        save_dict_train['f1'] = []
        save_dict_train['step_num'] = []        
    # Potentially calculate setMean to subtract
    if params['subtract_set_mean'] == 1:
        params['setMean'] = np.mean(params['images_means'][params['trainInd'],:],(0))
        print("Set Mean",params['setMean']) 
    
    # Meta scaler
    if params.get('meta_features',None) is not None and params['scale_features']:
        params['feature_scaler_meta'] = sklearn.preprocessing.StandardScaler().fit(params['meta_array'][params['trainInd'],:])  
        print("scaler mean",params['feature_scaler_meta'].mean_,"var",params['feature_scaler_meta'].var_) 
    
    params['trainSetState'] = 'train'
    balance_classes()
    get_loaders()
    initialize_model(cv)
    define_loss()
    define_optimizer()

    # Define softmax
    modelVars['softmax'] = nn.Softmax(dim=1)

    # loading from checkpoint
    start_epoch = restore_checkpoint(load_old)
        
    # History dictionary to store everything
    history = {
            'train_history_loss': [],
            'train_history_auc': [],
            'val_history_loss': [],
            'val_history_auc': [],
             }
    
    # Run training
    start_time = time.time()
    print("Start training...")
    
    tk0 = tqdm(range(start_epoch, params['training_steps']+1))
    for step in tk0: 
        train_loss,train_out,train_targets = train_fn()
        val_loss, outputs, targets = eval_fn(cv, step)
        
        duration = time.time() - start_time
        print("Config:",OUT)
        print('Fold: %d Epoch: %d/%d (%d h %d m %d s)' % (cv,step,params['training_steps'], int(duration/3600), int(np.mod(duration,3600)/60), int(np.mod(np.mod(duration,3600),60))) + time.strftime("%d.%m.-%H:%M:%S", time.localtime()))
        
        tk0.set_postfix(Train_Loss=train_loss, 
                        Valid_Loss=val_loss, 
                        ACC=save_dict['acc'],
                        F1=save_dict['f1'],
                        AUC=save_dict['auc'],
                        WAUC=save_dict['wauc'], 
                        Sensitivity=save_dict['sens'], 
                        Specificity=save_dict['spec'], 
                        Best=f"best WACC: {params['valBest']} at Epoch {params['lastBestInd']}"
                       )
        print("Confusion Matrix")
        print(conf_matrix)
        
        modelVars['scheduler'].step(save_dict['auc'])
        
        history['train_history_loss'].append(train_loss)
        history['train_history_auc'].append(train_auc)
        history['val_history_loss'].append(val_loss)
        history['val_history_auc'].append(auc_score)
        
    utils.print_history(cv,history,num_epochs=step+1)

         
                
    # Free everything in modelvars
    modelVars.clear()
    # After CV Training: print CV results and save them
    print("Best F1:",allData['f1Best'][cv])
    print("Best Sens:",allData['sensBest'][cv])
    print("Best Spec:",allData['specBest'][cv])
    print("Best Acc:",allData['accBest'][cv])
    print("Best Per Class Accuracy:",allData['waccBest'][cv])
    print("Best Weighted Acc:",np.mean(allData['waccBest'][cv]))
    print("Best AUC:",allData['aucBest'][cv])
    print("Best Mean AUC:",np.mean(allData['aucBest'][cv]))    
    print("Convergence Steps:",allData['convergeTime'][cv])

params['save_dir']



