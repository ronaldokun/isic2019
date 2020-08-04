from pathlib import Path
import imagesize
import numpy as np

BASE_PATH = Path.cwd().expanduser().parent

def ordered_crop(params):
    # Crop positions, always choose multiCropEval to be 4, 9, 16, 25, etc.
    params['cropPositions'] = np.zeros([len(params['img_paths']),params['multiCropEval'],2],dtype=np.int64)
    #params['imSizes'] = np.zeros([len(params['im_paths']),params['multiCropEval'],2],dtype=np.int64)
    for u, img in enumerate(params['img_paths']):
        height, width = imagesize.get(img)
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
    for u, img in enumerate(params['img_paths']):
        height_test, width_test = imagesize.get(img)
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
                
    return params   
