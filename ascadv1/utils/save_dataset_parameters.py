########################################
# Paths to dataset and folder have to be changed

import pickle
import numpy as np



dict_parameters = {}

dict_parameters['KEY'] = 0x00112233445566778899AABBCCDDEEFF 

########### PATHS & FILE MANAGEMENT ######################################
DATASET_FOLDER= ''  
FILE_DATASET = 'Ascad_v1_dataset.h5'
PROJECT_FOLDER = ''  

dict_parameters['DATASET_FOLDER'] = DATASET_FOLDER
dict_parameters['FILE_DATASET'] = FILE_DATASET
dict_parameters['PROJECT_FOLDER'] = PROJECT_FOLDER

###########################################################################

############### MASKING SCHEME ############################################

xor_mapping = np.empty((256,256,256),dtype = np.uint8)
for i in range(256):
    for j in range(256):
        xor_mapping[i,j,i^j] = 1
## Attacked intermediate states, '1' for first round
INTERMEDIATES = ['s1','t1','k1','p1']
ONE_MASK = ['o','r','i']
MASKS = ONE_MASK + ['ri','ro']
MASK_INTERMEDIATES = {'s1' : ['r','o','ro'],'t1':['r','i','ri'],'k1':['o','ro','r'],'p1': []}
VARIABLE_LIST = {}
MASKED_INTERMEDIATES = []

for intermediate in INTERMEDIATES:
  
    VARIABLE_LIST[intermediate] = [ intermediate[:len(list(intermediate))-1] + '0'+ ('0'+str(i) if i < 10 else '' + str(i)) for i in range(1 if int(intermediate[len(list(intermediate))-1]) == 1 else 17,17 if int(intermediate[len(list(intermediate))-1]) == 1 else 33 )  ]
    for mask in MASK_INTERMEDIATES[intermediate]:
        MASKED_INTERMEDIATES.append(intermediate + '^' + mask)
        start_byte = 1 
        end_byte= 17
        if not intermediate + '^' + mask in VARIABLE_LIST:
                VARIABLE_LIST[intermediate + '^' + mask] = []             
        for byte in range(start_byte,end_byte):      
            mask_name = mask if mask == 'o' or mask == 'i' else ((mask + '0' + str(byte)) if byte < 10 else mask + str(byte))
            VARIABLE_LIST[intermediate + '^' + mask].append((intermediate[:len(list(intermediate))-1] + '0'+ ('0'+str(byte) if byte < 10 else '' + str(byte) ) +'^' +  mask_name))  


########################################################################                        

INTERMEDIATES += MASKS
INTERMEDIATES += MASKED_INTERMEDIATES
        
for intermediate in MASKS:
    VARIABLE_LIST[intermediate] = [intermediate] if (intermediate == 'o' or intermediate == 'i')else [(intermediate + '0' + str(byte) if byte < 10 else (intermediate + str(byte))) for byte in range( 1, 17)]



dict_parameters['INTERMEDIATES'] = INTERMEDIATES
dict_parameters['ONE_MASKS'] = ONE_MASK
dict_parameters['MASKS'] = MASKS
dict_parameters['MASK_INTERMEDIATES'] = MASK_INTERMEDIATES
dict_parameters['MASKED_INTERMEDIATES'] = MASKED_INTERMEDIATES
dict_parameters['VARIABLE_LIST'] = VARIABLE_LIST

file = open('dataset_parameters','wb')
pickle.dump(dict_parameters,file)
file.close()