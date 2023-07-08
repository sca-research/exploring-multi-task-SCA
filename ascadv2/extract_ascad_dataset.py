# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:58:12 2022

@author: martho
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 26 10:28:59 2022

@author: martho
"""

import h5py
import numpy as np
import pickle
from utils.generate_intermediate_values import save_real_values
from utility import normalise_traces_to_int8 , TRACES_FOLDER , DATASET_FOLDER

from tqdm import tqdm
import argparse



def create_dataset_raw():
    FOLDER_TRACES = TRACES_FOLDER
   
    
    FOLDER = DATASET_FOLDER
    REAL_VAL_FOLDER = FOLDER + 'real_values/'
    
    indexes_rin = np.load(FOLDER_TRACES + 'extraction_rin.npy') 
    indexes_alpha = np.load(FOLDER_TRACES + 'extraction_alpha.npy') 
    indexes_beta= np.load(FOLDER_TRACES + 'extraction_beta.npy') 
    indexes_mj = []
    indexes_m = []
    indexes_s1_mj = []
    indexes_t1_mj = []
    for byte in range(16):
        indexes_mj.append(np.load(FOLDER_TRACES + 'extraction_{}_{}_block2.npy'.format('mj',byte))) 
        indexes_m.append(np.concatenate([np.load(FOLDER_TRACES + 'extraction_{}_{}_block1.npy'.format('m',byte)), np.load(FOLDER_TRACES + 'extraction_{}_{}_block2.npy'.format('m',byte))]))
        indexes_s1_mj.append(np.load(FOLDER_TRACES + 'extraction_{}_{}.npy'.format('s1^mj',byte))) 
        indexes_t1_mj.append(np.load(FOLDER_TRACES + 'extraction_{}_{}.npy'.format('t1^mj',byte))) 
    indexes_all_besides_round_block = np.concatenate([indexes_alpha,indexes_rin,indexes_beta,np.array(indexes_m).reshape(-1),np.array(indexes_mj).reshape(-1),np.array(indexes_s1_mj).reshape(-1),np.array(indexes_t1_mj).reshape(-1)])
    print(indexes_all_besides_round_block.shape)
        
        
    

    traces = np.empty((800000,4304+93*16*2),dtype = np.int8)
    coefficient  = 4
    start = int(455354//coefficient)
    start_permutations = 111907
    step_round = int(45266//coefficient) 
    step_byte = int(373 //coefficient)
    rest_start  = 455354 / coefficient - start
    rest_start_permutations  = 455354 / coefficient - start_permutations
    rest_round  = 45266 / coefficient - step_round
    rest_byte  = 373 / coefficient - step_byte
    indexes_points = []
    indexes_points_permutations = []
    for round_k in range(0,10):
        for byte in range(0,16):
            beta = round(rest_start + rest_byte * byte + rest_round* round_k)    
            for points in range(step_byte):
                if round_k == 0:
                    indexes_points.append(start+ step_round * (round_k) + step_byte * (byte) + beta + points)
                indexes_points_permutations.append(start+ step_round * (round_k) + step_byte * (byte) + beta + points)
           

    for i in tqdm(range(8)):
        trace = np.load(FOLDER_TRACES + 'traces_{}.npy'.format(i+1),allow_pickle = True)
        intermediate_points = trace[:,indexes_points]
        permutations_points = trace[:,indexes_points_permutations]
        permutations_points = permutations_points.reshape((100000,10,step_byte * 16))
        permutations_points = np.mean(permutations_points,axis = 1).astype(np.int8)
       
        
        
        traces[100000*i:100000*(i+1)] = np.concatenate([trace[:,indexes_all_besides_round_block],intermediate_points,permutations_points],axis = 1)

    n_profiling = 450000
    n_test = 45000
    n_attack = 5000
    n_files = [1]
    master_key =[0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                      0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF ]  
    
   
    fake_key =  0x00112233445566778899AABBCCDDEEFF
    traces_index = np.random.choice(range(0,800000),size = 500000,replace = False)
    indexes = {}
    indexes['training'] = traces_index[:n_profiling]
    indexes['validation'] = traces_index[n_profiling:n_profiling+n_test]
    indexes['attack']= traces_index[n_profiling+n_test:n_profiling+n_test+n_attack]
    
    assert set(indexes['training']).isdisjoint(set(indexes['validation'] ))
    assert set(indexes['training']).isdisjoint(set(indexes['attack']))
    assert set(indexes['validation']).isdisjoint(set(indexes['attack']))
    
    

    
    
    h5_file = h5py.File(FOLDER + 'Ascad_v2_fully_extracted.h5','w')
    datasets = ['training','validation','attack']
    keys = np.load(REAL_VAL_FOLDER + 'keys.npy')
    plaintexts = np.load(REAL_VAL_FOLDER + 'plaintexts.npy')
    masks = np.load(REAL_VAL_FOLDER + 'masks.npy')
    
    
    
    for dataset in datasets:
        group = h5_file.create_group(dataset)
        print(dataset)
        items_to_add = {}
        items_to_add['traces'] = traces[indexes[dataset]]
        items_to_add['masks'] = masks[indexes[dataset]]
        # items_to_add['ciphertexts'] =ciphertexts[indexes[dataset]]
                 
        items_to_add['keys'] = keys[indexes[dataset]]
        items_to_add['plaintexts'] = plaintexts[indexes[dataset]]    

        
        labels_dict = save_real_values( plaintexts= items_to_add['plaintexts'], random =items_to_add['masks'],keys=items_to_add['keys'],save_file = False)
        # _ , labels_dict, _ = read_from_h5_file(n_traces = 200000 if dataset == 'training' else 50000,dataset = dataset,file = 'Ascad_v2_dataset_extracted_aihws.h5')
        
        labels = group.create_group('labels')
        
        for item , data in labels_dict.items():
            labels.create_dataset(item,data= data,compression = 'gzip') 

        for item , data in items_to_add.items():
            group.create_dataset(item,data= data,compression = 'gzip')        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform correlations in order to find PoIs')
    parser.add_argument('-n', action="store", dest="N", help='Use trs file given as argument', type=int, default=1)
    parser.add_argument('--REGROUP', action="store_true", dest="REGROUP",
                        help='Add leakage of masks',  default=False)
    args  = parser.parse_args()
    N = args.N
    REGROUP = args.REGROUP


    
    create_dataset_raw()
    



        
 
    
    
    
