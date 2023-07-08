# -*- coding: utf-8 -*-

import argparse
import parse
import os
import numpy as np



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from multiprocessing import Process
from train_models import model_single_task_xor , model_multi_task , model_single_task_twin

# import dataset paths and variables
from utility import   MODEL_FOLDER

# import custom layers

from utility import load_dataset, load_dataset_multi ,load_model_from_name


#### Training high level function
def test_model(convolution_blocks , kernel_size,filters, strides , pooling_size,dense_blocks,dense_units,model_type):
    id_model  = 'cb{}ks{}f{}s{}ps{}db{}du{}'.format(convolution_blocks , kernel_size,filters,strides , pooling_size,dense_blocks,dense_units)

    n_traces = 10000
    if 'multi_task' in model_type:
        multi_name = 'model_{}_all_{}.h5'.format(model_type,id_model) 

        model_struct = model_multi_task(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,input_length = 250000,summary = False)
        model = load_model_from_name(model_struct,multi_name)  
        X_profiling = load_dataset_multi(n_traces = n_traces,dataset = 'attack')
    else:
        for byte in range(6,7):
            name = 'model_{}_{}_{}.h5'.format(model_type,byte,id_model) 
            if 'xor' in name:
                model_struct = model_single_task_xor(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,input_length = 250000,summary = False)                
            else:
                model_struct = model_single_task_twin(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,input_length = 250000,summary = False)    
            model = load_model_from_name(model_struct,name) 
    
        X_profiling = load_dataset(6,n_traces = n_traces,dataset = 'attack')
    results = model.evaluate(X_profiling[0],X_profiling[1], batch_size=256)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--XOR',   action="store_true", dest="SINGLE_TASK_XOR", help='Adding the masks to the labels', default=False)
    parser.add_argument('--TWIN',   action="store_true", dest="SINGLE_TASK_TWIN", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI',   action="store_true", dest="MULTI", help='Adding the masks to the labels', default=False)
    #parser.add_argument('-scenario',   action="store", dest="TRAINING_TYPE", help='Adding the masks to the labels', default='extracted')
    parser.add_argument('--ALL',   action="store_true", dest="ALL", help='Adding the masks to the labels', default=False)
        
    args            = parser.parse_args()
  

   
    SINGLE_TASK_XOR        = args.SINGLE_TASK_XOR
    SINGLE_TASK_TWIN = args.SINGLE_TASK_TWIN
    MULTI = args.MULTI
    ALL = args.ALL
    #TRAINING_TYPE= args.TRAINING_TYPE

    TARGETS = {}

    if SINGLE_TASK_XOR:
        MODEL_TYPE = ['single_task_xor']
    elif SINGLE_TASK_TWIN:
        MODEL_TYPE = ['single_task_twin']
    elif MULTI:
        MODEL_TYPE = ['multi_task']
    elif ALL:
        MODEL_TYPE = ['single_task_xor','single_task_twin','multi_task']
    else:
        print('No training mode selected')

    for model_type in MODEL_TYPE:
        for model_name in os.listdir(MODEL_FOLDER):
            byte = 'all' if 'multi' in model_name else '6'
            multi_task =  'model_{}_{}'.format(model_type,byte)
            print(model_name)
            if not multi_task  in model_name :
                continue
            
            format_string = multi_task + '_cb{}ks{}f{}s{}ps{}db{}du{}.h5'
            parsed = parse.parse(format_string,model_name)
            convolution_blocks = int(parsed[0])
            kernel_size_list = parsed[1][1:-1]
            kernel_size_list = kernel_size_list.split(',')   
            kernel_size = [int(elem) for elem in kernel_size_list]
            filters = int(parsed[2])
            strides = int(parsed[3])
            pooling_size = int(parsed[4])
            dense_blocks = int(parsed[5])
            dense_units = int(parsed[6])

    
            process_eval = Process(target=test_model, args=(convolution_blocks , kernel_size,filters,strides , pooling_size,dense_blocks,dense_units,model_type))
            process_eval.start()
            process_eval.join()


    print("$ Done !")