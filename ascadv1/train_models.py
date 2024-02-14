import argparse
import os
import numpy as np
import pickle
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Softmax, Dense, Input,Conv1D, AveragePooling1D, BatchNormalization , Concatenate 
from tensorflow.keras.optimizers import Adam
from multiprocessing import Process


# import dataset paths and variables
from utility import  METRICS_FOLDER , MODEL_FOLDER

# import custom layers
from utility import XorLayer , PoolingCrop  , SharedWeightsDenseLayer

from utility import load_dataset, load_dataset_multi 

import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()
seed = 42


tf.random.set_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

  

###########################################################################
# ALL ASCAD-R experiments 
#   model single-task m_s

def model_single_task(convolution_blocks = 1,dense_blocks =2, kernel_size = [34],filters = 4, strides = 17 , pooling_size = 2,dense_units= 200,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = True,seed = 42):
    inputs_dict = {}
    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
    
    branch = input_layer_creation(inputs,input_length,seed = seed)
    branch = cnn_core(branch,convolution_blocks = 1, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)

    outputs = {}
      
    mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True,seed = seed)
    intermediate_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True,seed = seed)
    
    xor = XorLayer(name ='output' )([mask_branch,intermediate_branch]) 
    outputs['output'] = xor
    
    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_single_task_xor_{}'.format(name))

    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if summary:
        model.summary() 
    return model


##############################################################
#  Section 5.1 Leveraging common masks with a shared branch : 
#   model multi-task m_{nt + d + 1} 

def model_multi_task_single_target_one_shared_mask(convolution_blocks = 1,dense_blocks =2, kernel_size = [34],filters = 4, strides = 17 , pooling_size = 2,dense_units= 200,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False,seed = 42):
    
    inputs_dict = {}    
    outputs = {} 
    metrics = {}
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length,seed = seed)
    branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)

    mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True,seed = seed)



    

    for byte in range(2,16):
        
        intermediate_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = False,seed = seed)
        xor = XorLayer(name ='xor_{}'.format(byte) )([intermediate_branch,mask_branch])        
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_t_{}'.format(byte))(xor)
        metrics['output_{}'.format(byte)] = 'accuracy'
        
    losses = {}   
    




    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')        

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer,metrics = metrics)
    
    if summary:
        model.summary()
    return model   
##############################################################
#  Section 5.1 Leveraging common masks with a shared branch : 
#   model multi-task m_{d} 

def model_multi_task_single_target_one_shared_mask_shared_branch(convolution_blocks = 1,dense_blocks =2, kernel_size = [34],filters = 4, strides = 17 , pooling_size = 2,dense_units= 200,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False,seed = 42):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length,seed = seed)
    branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)

    mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True,seed = seed)
    shared_branch_t_i = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14,seed = seed)

    outputs = {} 

    for byte in range(2,16):
        
        xor = XorLayer(name ='xor_{}'.format(byte) )([shared_branch_t_i[:,:,byte-2],mask_branch])        
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_{}'.format(byte))(xor)
        
    losses = {}   
    



    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')   
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy']) 
    if summary:
        model.summary()
    return model   


##############################################################
#  Section 5.2 Leveraging different masks using low-level parameter sharing : 
#   model multi-task m_{d} 

def model_multi_task_single_target(convolution_blocks = 1,dense_blocks =2, kernel_size = [34],filters = 4, strides = 17 , pooling_size = 2,dense_units= 200,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False,seed = 42):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length,seed = seed)
    branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)

    shared_branch_s_r = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14,seed = seed)
    shared_branch_r = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14,seed = seed)
    
    outputs = {} 
    
    for byte in range(2,16):
        
        shared_branch_r_activated = Softmax()(shared_branch_r[:,:,byte-2])   
        
        xor = XorLayer(name ='xor_{}'.format(byte) )([shared_branch_s_r[:,:,byte-2],shared_branch_r_activated])        
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_{}'.format(byte))(xor)
        
    losses = {}   
    



    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')      
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy']) 
    if summary:
        model.summary()
    return model   

##############################################################
#  Section 5.2 Leveraging different masks using low-level parameter sharing : 
#   model multi-task m_{nt * d} 

def model_multi_task_single_target_not_shared(convolution_blocks = 1,dense_blocks =2, kernel_size = [34],filters = 4, strides = 17 , pooling_size = 2,dense_units= 200,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False,seed = 42):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length,seed = seed)
    branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size,seed = seed)


    outputs = {} 
    
    for byte in range(2,16):
        
        intermediate_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = False,seed = seed)
        mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True,seed = seed)
                
        xor = XorLayer(name ='xor_{}'.format(byte) )([intermediate_branch,mask_branch])        
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_{}'.format(byte))(xor)
    losses = {}   
    



    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy']) 
    if summary:
        model.summary()
    return model   






################################################################
# ALL ASCAD-R EXPERIMENTS
# block of layers noted as \theta_{\forall} 

def input_layer_creation(inputs,input_length,target_size = 25000,seed = 42,name = ''):

    size = input_length
    
    iteration  = 0
    crop = inputs
    
    while size > target_size:
        crop = PoolingCrop(input_dim = size,name = name,seed = seed)(crop)
        iteration += 1
        size = math.ceil(size/2)

    x = crop  
    return x

def cnn_core(inputs_core,convolution_blocks , kernel_size,filters, strides , pooling_size,seed = 42):
    x = inputs_core
    for block in range(convolution_blocks):
        x = Conv1D(kernel_size=(kernel_size[block],), strides=strides, filters=filters, activation='selu', padding='same',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)    
        x = BatchNormalization()(x)
        x = AveragePooling1D(pool_size = pooling_size)(x)
        
    output_layer = Flatten()(x) 

    return output_layer


################################################################
# ALL ASCAD-R EXPERIMENTS
# prediction heads of models without low-level parameter sharing

def dense_core(inputs_core,dense_blocks,dense_units,activated = False,seed = 42):
    x = inputs_core    
    for block in range(dense_blocks):
        x = Dense(dense_units, activation='selu',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)
    if activated:
        output_layer = Dense(256,activation ='softmax' ,kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)  
    else:
        output_layer = Dense(256,kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)   
    return output_layer    


################################################################
# ALL ASCAD-R EXPERIMENTS
# prediction heads of models with low-level parameter sharing

def dense_core_shared(inputs_core, shared_block = 1,non_shared_block = 1, units = 64, branches = 14,seed = 42):
    non_shared_branch = []
    for branch in range(branches):
        x = inputs_core
        for block in range(non_shared_block):
            x = Dense(units,activation ='selu',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)
        non_shared_branch.append(tf.expand_dims(x,2))
    x = Concatenate(axis = 2)(non_shared_branch)
   
    for block in range(shared_block):
        x = SharedWeightsDenseLayer(input_dim = x.shape[1],units = units,shares = 14,seed = seed)(x)        
    output_layer = SharedWeightsDenseLayer(input_dim = x.shape[1],units = 256,activation = False,shares = 14,seed = seed)(x)   
    return output_layer 

################################################################
#### Training high level function #####
def train_model(training_type,byte,seed):
    epochs = 100
    batch_size = 250 
    n_traces = 50000
    single_task = False
    if 'single_task' in training_type:
        single_task = True
        X_profiling , validation_data = load_dataset(byte,target = 't1' if 'subin' in training_type else 's1',n_traces = n_traces,dataset = 'training')
        model_t = 'model_{}'.format(training_type) 
    elif ('multi_task_single_target_one_shared_mask' in training_type) or ('multi_task_single_target_multi_shares' in training_type) :
        X_profiling , validation_data = load_dataset_multi('t1',n_traces = n_traces,dataset = 'training') 
        model_t = 'model_{}'.format(training_type)     
        
    elif 'multi_task_single_target' in training_type:
        X_profiling , validation_data = load_dataset_multi('s1',n_traces = n_traces,dataset = 'training') 
        model_t = 'model_{}'.format(training_type)
        
    else:
        X_profiling , validation_data = load_dataset_multi('t1',n_traces = n_traces,dataset = 'training') 
        model_t = 'model_multi_task'
    
    window =  X_profiling.element_spec[0]['traces'].shape[0]
    monitor = 'val_accuracy'
    mode = 'max'

    # m_s
    if single_task:
        
        model = model_single_task(input_length = window)       
    
    # m_d  SBOX OUTPUT
    elif model_t == 'model_multi_task_single_target':
        model = model_multi_task_single_target(input_length = window,seed = seed)         
        monitor = 'val_loss'   
        mode = 'min'    
    # m_{nt * d}   SBOX OUTPUT
    elif model_t == 'model_multi_task_single_target_not_shared':

        model = model_multi_task_single_target_not_shared(input_length = window,seed = seed)         
        monitor = 'val_loss'   
        mode = 'min'     
    # m_{nt + d - 1}  SBOX INPUT 
    elif model_t == 'model_multi_task_single_target_one_shared_mask':
        
        model = model_multi_task_single_target_one_shared_mask(input_length = window)                  
        monitor = 'val_loss'   
        mode = 'min'
    # m_d  SBOX INPUT
    elif model_t == 'model_multi_task_single_target_one_shared_mask_shared_branch':
        
        model = model_multi_task_single_target_one_shared_mask_shared_branch(input_length = window,seed = seed)                  
        monitor = 'val_loss'   
        mode = 'min'             
    else:
        print('Some error here')

    
    X_profiling = X_profiling.shuffle(len(X_profiling)).batch(batch_size) 
    validation_data = validation_data.batch(batch_size)
    file_name = '{}_{}'.format(model_t,byte) 
    callbacks = tf.keras.callbacks.ModelCheckpoint(
                                filepath= MODEL_FOLDER+ file_name+'.h5',
                                save_weights_only=True,
                                monitor=monitor,
                                mode=mode,
                                save_best_only=True)

    

    
    history = model.fit(X_profiling, batch_size=batch_size, verbose = 1, epochs=epochs, validation_data=validation_data,callbacks =callbacks)

    print('Saved model {} ! '.format(file_name))
 
    file = open(METRICS_FOLDER+'history_training_'+(file_name ),'wb')
    pickle.dump(history.history,file)
    file.close()
    return model






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    args            = parser .parse_args()
  

 
    TARGETS = {}
    
    training_types = ['single_task_subout','single_task_subin']



    seeds_random = np.random.randint(0,9999,size = 9)
    # Because 42
    seeds_random = np.concatenate([[42],seeds_random],axis = 0)
    
    for seed in seeds_random:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        for training_type in training_types:
            
            # Depending on your setup, you might need to remove the "Process"
            
            if not 'single_task' in training_type:
                process_eval = Process(target=train_model, args=(training_type,'all',seed))
                process_eval.start()
                process_eval.join()      
            else:
                for byte in range(2,16):       
                    process_eval = Process(target=train_model, args=(training_type,byte,seed))
                    process_eval.start()
                    process_eval.join()       
                    
            # if not 'single_task' in training_type:
            #     train_model(training_type,'all',seed)

            # else:
            #     for byte in range(2,16):       
            #         train_model(training_type,byte,seed)
                    
                      

    print("$ Done !")
            
        
        
