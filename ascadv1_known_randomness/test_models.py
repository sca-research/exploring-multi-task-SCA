# -*- coding: utf-8 -*-

import argparse

import os
import numpy as np



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from multiprocessing import Process
from train_models import cnn_best , cnn_multi_task_multi_target , cnn_multi_task_subbytes_inputs , cnn_hierarchical_multi_target, cnn_hierarchical_subbytes_inputs

# import dataset paths and variables
from utility import    VARIABLE_LIST , METRICS_FOLDER

# import custom layers

from utility import load_dataset, load_dataset_multi ,load_dataset_hierarchical ,load_model_from_name


#### Training high level function
def test_model( training_type,target_byte ,target):

    if training_type  == 'classical':
        model = cnn_best(input_length=250000)
        model = load_model_from_name(model,'{}_cnn_single.h5'.format(target_byte)) 
        X , Y = load_dataset(target_byte,target,VARIABLE_LIST[target].index(target_byte),n_traces = 10000,dataset = 'attack')
        predictions = model.evaluate(X, Y,batch_size = 250)
        np.save(METRICS_FOLDER + target_byte + '_classical_accuracy.npy',predictions[1])
    if training_type == 'multi':
        model = cnn_multi_task_subbytes_inputs(input_length=250000,summary = False) if target == 't1' else cnn_multi_task_multi_target(input_length=250000,summary = False)
        model_type = 'multi_task_subbytes_inputs'  if target == 't1' else 'multi_task_multi_target'
        model = load_model_from_name(model,'{}_cnn_{}.h5'.format(target_byte,model_type)) 
        X , Y = load_dataset_multi(VARIABLE_LIST[target].index(target_byte),n_traces = 10000,dataset = 'attack',multi_target = target == 'k1')
        predictions = model.evaluate(X, Y,batch_size = 250)    
        np.save(METRICS_FOLDER + target_byte +'_multi_accuracy.npy',predictions[5:] )
    if training_type == 'multi':
        model = cnn_hierarchical_subbytes_inputs(input_length=250000,summary = False) if target == 't1' else cnn_hierarchical_multi_target(input_length=250000,summary = False)
        model_type = 'hierarchical_subbytes_inputs' if target == 't1' else 'hierarchical_multi_target'
        model = load_model_from_name(model,'{}_cnn_{}.h5'.format(target_byte,model_type)) 
        X , Y = load_dataset_hierarchical(VARIABLE_LIST[target].index(target_byte),n_traces = 10000,dataset = 'attack',multi_target = target == 'k1')
        predictions = model.evaluate(X, Y,batch_size = 250)    
        np.save(METRICS_FOLDER + target_byte +'_hierarchical_accuracy.npy',predictions[7:] )       
            
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')

    parser.add_argument('--CLASSICAL', action="store_true", dest="CLASSICAL",
                        help='Classical training of the intermediates', default=False)
    parser.add_argument('--MULTI',   action="store_true", dest="MULTI", help='Adding the masks to the labels', default=False)
    parser.add_argument('--HIERARCHICAL',   action="store_true", dest="HIERARCHICAL", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI_TARGET',   action="store_true", dest="MULTI_TARGET", help='Adding the masks to the labels', default=False)
    args            = parser.parse_args()
  

    HIERARCHICAL        = args.HIERARCHICAL
    CLASSICAL        = args.CLASSICAL
    MULTI = args.MULTI
    MULTI_TARGET = args.MULTI_TARGET
    TARGETS = {}
    if CLASSICAL:   
       training_types = ['classical']
       TARGETS['classical'] = ['i' , 's1^r','t1^r','r','t1^i','t1^ri'] 
       # TARGETS['classical'] = ['i'] 
       BYTES = [i for i in range(2,16)]
    elif MULTI:
        training_types = ['multi']
        TARGETS['multi'] = ['t1'] if not MULTI_TARGET else ['k1']
        BYTES = [i for i in range(2,16)]

    elif HIERARCHICAL:
        training_types = ['hierarchical']
        TARGETS['hierarchical'] = ['t1'] if not MULTI_TARGET else ['k1']
        BYTES = [i for i in range(2,16)]
        #BYTES = [4,5,6,7,9,10,11,12,13,15,16]
    
    else:
        print('No training mode selected')
        training_type = 'multi'


    for training_type in training_types:
        for TARGET in TARGETS[training_type]:
            acc_target = []
            if not TARGET == 'i':
                for BYTE in BYTES:
                    
                    target_byte = VARIABLE_LIST[TARGET][BYTE] 
            #         acc = np.load(METRICS_FOLDER + target_byte + '_classical_accuracy.npy')
            #         acc_target.append(acc)
            #     print(TARGET, np.mean(acc_target))
            # else:
            #     acc = np.load(METRICS_FOLDER + 'i' + '_classical_accuracy.npy')
            #     print('i :',acc)
                    process_eval = Process(target=test_model, args=(training_type,target_byte ,TARGET))
                    process_eval.start()
                    process_eval.join()
            else:
                process_eval = Process(target=test_model, args=( training_type,'i' ,TARGET))
                process_eval.start()
                process_eval.join()


    print("$ Done !")