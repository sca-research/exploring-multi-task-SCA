# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:10:35 2021

@author: martho
"""


import argparse
import os
import numpy as np
import pickle
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Softmax, Dense, Input,Conv1D, AveragePooling1D, BatchNormalization,Add , Concatenate , Multiply
from tensorflow.keras.optimizers import Adam
from multiprocessing import Process


# import dataset paths and variables
from utility import  METRICS_FOLDER , MODEL_FOLDER

# import custom layers
from utility import XorLayer , PoolingCrop , InvSboxLayer , SharedWeightsDenseLayer

from utility import load_dataset, load_dataset_multi 
from tqdm import tqdm


seed = 42


tf.random.set_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'



    
class Multi_Model(tf.keras.Model):
    
    def __init__(self,inputs,outputs):
        
        super(Multi_Model, self).__init__(inputs = inputs,outputs = outputs)
        
        all_maps = np.load('xor_mapping.npy')
        mapping1 = []
        mapping2 = []
        for classe in range(256):
            mapped = np.where(all_maps[classe] == 1)
            mapping1.append(mapped[0])
            mapping2.append(mapped[1])
        self.xor_mapping1 = np.array(mapping1)
        self.xor_mapping2 = np.array(mapping2)
        self.loss_tracker = tf.keras.metrics.Mean(name= 'regularized_loss')        


    
    def train_step(self, data):
        x, y = data
   
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(
                y,
                y_pred,
                regularization_losses=self.losses,
            )
            losses = {}
            sum_losses = 0
            for byte in range(2,16):
                losses[byte] = tf.keras.losses.categorical_crossentropy(y['output_{}'.format(byte)], y_pred['output_{}'.format(byte)])
                sum_losses = sum_losses + losses[byte]
            sum_losses = sum_losses / 14
            for byte in range(2,16):
            
            
                loss = loss + tf.math.pow(losses[byte] - sum_losses,2) / 14
            loss = loss + sum_losses
            
                
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        dict_metrics = {}
        for m in self.metrics:
            dict_metrics[m.name] = m.result()
        dict_metrics['loss'] = self.loss_tracker.result()

        return dict_metrics

    
    def test_step(self,data):
        x, y = data
        # forward pass, no backprop, inference mode 
        y_pred = self(x, training=False)
        loss = self.compiled_loss(
             y,
             y_pred,
             regularization_losses=self.losses,
        )
         
        losses = {}
        sum_losses = 0
        for byte in range(2,16):
            losses[byte] = tf.keras.losses.categorical_crossentropy(y['output_{}'.format(byte)], y_pred['output_{}'.format(byte)])
            sum_losses = sum_losses + losses[byte]
        sum_losses = sum_losses / 14
        for byte in range(2,16):
        
        
            loss = loss + tf.math.pow(losses[byte] - sum_losses,2) / 14
        loss = loss + sum_losses
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        dict_metrics = {}
        for m in self.metrics:
            dict_metrics[m.name] = m.result()
        dict_metrics['loss'] = self.loss_tracker.result()
        return dict_metrics
        


###########################################################################

def model_single_task(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,mlp = False,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = True):
    inputs_dict = {}
    outputs = {}
    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
    branch = input_layer_creation(inputs,input_length)

    if not mlp :
        branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size)
    else:
        branch = branch[:,:,0]
    
      
    branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = False)
    outputs['output'] = Softmax(name ='output')(branch)
    
    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_single_task_xor_{}'.format(name))

    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if summary:
        model.summary() 
    return model




def model_multi_task(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,mlp = False,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False):
    
    inputs_dict = {}    
    outputs = {}
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    #branch = input_layer_creation(inputs,input_length)
    if not mlp:
        branch = cnn_core(inputs,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size)
    else:
        branch = branch[:,:,0]
        
    branch_i = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True)   
    outputs['output_i'] = branch_i
    shared_branch_t_i = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14)
    shared_branch_t_r = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14)
    shared_branch_t_ri = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14)
    shared_branch_r = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14)

    

    for byte in range(2,16):
        shared_branch_r_activated = Softmax(name = 'activated_mask_r_{}'.format(byte))(shared_branch_r[:,:,byte-2])
        outputs['output_r_{}'.format(byte)] = shared_branch_r_activated
        shared_branch_t_i_activated = Softmax(name = 'activated_t_i_{}'.format(byte))(shared_branch_t_i[:,:,byte-2])
        outputs['output_t_i_{}'.format(byte)] = shared_branch_t_i_activated
        shared_branch_t_r_activated = Softmax(name = 'activated_t_r_{}'.format(byte))(shared_branch_t_r[:,:,byte-2])
        outputs['output_t_r_{}'.format(byte)] = shared_branch_t_r_activated
        shared_branch_t_ri_activated = Softmax(name = 'activated_t_ri_{}'.format(byte))(shared_branch_t_ri[:,:,byte-2])
        outputs['output_t_ri_{}'.format(byte)] = shared_branch_t_ri_activated        
        
        branch_ri = XorLayer(name ='xor_{}'.format(byte) )([shared_branch_r_activated,branch_i])
       
        xor_i = XorLayer(name ='xor_i_{}'.format(byte) )([shared_branch_t_i[:,:,byte-2],branch_i])        
        xor_r = XorLayer(name ='xor_r_{}'.format(byte) )([shared_branch_t_r[:,:,byte-2],shared_branch_r_activated])  
        xor_ri = XorLayer(name ='xor_ri_{}'.format(byte) )([shared_branch_t_ri[:,:,byte-2],branch_ri])  

        add_shares = Add(name = 'add_shares_{}'.format(byte))([xor_i,xor_r,xor_ri])
        
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_{}'.format(byte))(add_shares)
        
    losses = {}   
    



    if not multi_model:
        for k , v in outputs.items():
            losses[k] = 'categorical_crossentropy'


        model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')  
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'])
    else:
        model = Multi_Model(inputs = inputs_dict,outputs = outputs)  
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, metrics=['accuracy'])        
    if summary:
        model.summary()
    return model   


def model_target(input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False):
    
    inputs_dict = {}    
    outputs = {} 
    metrics = {}
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length)
    branch = branch[:,:,0]

    

    branch_i = dense_core(branch,dense_blocks = 2,dense_units = 128,batch_norm = True,activated = True)   
    outputs['output_i'] = branch_i
    
    shared_branch_t_i = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 128,branches = 14)
    shared_branch_t_r = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 128,branches = 14)
    shared_branch_t_ri = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 128,branches = 14)
    
    shared_branch_s_r = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 128,branches = 14)
    shared_branch_r = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 128,branches = 14)

    

    for byte in range(2,16):
        shared_branch_r_activated = Softmax(name = 'activated_mask_r_{}'.format(byte))(shared_branch_r[:,:,byte-2])
        outputs['output_r_{}'.format(byte)] = shared_branch_r_activated
        shared_branch_t_i_activated = Softmax(name = 'activated_t_i_{}'.format(byte))(shared_branch_t_i[:,:,byte-2])
        outputs['output_t_i_{}'.format(byte)] = shared_branch_t_i_activated
        shared_branch_s_r_activated = Softmax(name = 'activated_s_r_{}'.format(byte))(shared_branch_s_r[:,:,byte-2])
        outputs['output_s_r_{}'.format(byte)] = shared_branch_s_r_activated
        shared_branch_t_r_activated = Softmax(name = 'activated_t_r_{}'.format(byte))(shared_branch_t_r[:,:,byte-2])
        outputs['output_t_r_{}'.format(byte)] = shared_branch_t_r_activated
        shared_branch_t_ri_activated = Softmax(name = 'activated_t_ri_{}'.format(byte))(shared_branch_t_ri[:,:,byte-2])
        outputs['output_t_ri_{}'.format(byte)] = shared_branch_t_ri_activated
        
        
        ri = XorLayer(name ='xor_i_{}'.format(byte) )([shared_branch_r_activated,branch_i])   
        
        xor_tr = XorLayer(name ='xor_tr_{}'.format(byte) )([shared_branch_t_r[:,:,byte-2],shared_branch_r_activated])       
        xor_tri = XorLayer(name ='xor_tri_{}'.format(byte) )([shared_branch_t_ri[:,:,byte-2],ri])          
        xor_ti = XorLayer(name ='xor_ti_{}'.format(byte) )([shared_branch_t_i[:,:,byte-2],branch_i])       
        xor_sr = XorLayer(name ='xor_sr_{}'.format(byte) )([shared_branch_s_r[:,:,byte-2],shared_branch_r_activated])  
        inv_sbox = InvSboxLayer(name = 'inv_sbox_layer_{}'.format(byte))(xor_sr)
        
        add_shares = Add(name = 'add_shares_{}'.format(byte))([inv_sbox,xor_ti,xor_tr,xor_tri])
        
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_{}'.format(byte))(add_shares)
        metrics['output_{}'.format(byte)] = 'accuracy'
        
    losses = {}   
    



  
    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')  
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=metrics)

    return model   




def input_layer_creation(inputs,input_length,target_size = 25000,name = ''):

    size = input_length
    
    iteration  = 0
    crop = inputs
    
    while size > target_size:
        crop = PoolingCrop(input_dim = size,name = name)(crop)
        iteration += 1
        size = math.ceil(size/2)

    x = crop  
    return x



### Cnn for shared layers and mask/permutations single task models.

def cnn_core(inputs_core,convolution_blocks , kernel_size,filters, strides , pooling_size):
    x = inputs_core
    for block in range(convolution_blocks):
        x = Conv1D(kernel_size=(kernel_size[block],), strides=strides, filters=filters, activation='selu', padding='same',kernel_initializer=tf.keras.initializers.RandomUniform(seed=7))(x)    
        x = BatchNormalization()(x)
        x = AveragePooling1D(pool_size = pooling_size)(x)
        
    output_layer = Flatten()(x) 

    return output_layer

def dense_core(inputs_core,dense_blocks,dense_units,batch_norm = False,activated = False):
    x = inputs_core    
    for block in range(dense_blocks):
        x = Dense(dense_units, activation='selu',kernel_initializer=tf.keras.initializers.RandomUniform(seed=7))(x)
        if batch_norm:
           x = BatchNormalization()(x)
    if activated:
        output_layer = Dense(256,activation ='softmax' ,kernel_initializer=tf.keras.initializers.RandomUniform(seed=7))(x)  
    else:
        output_layer = Dense(256,kernel_initializer=tf.keras.initializers.RandomUniform(seed=7))(x)   
    return output_layer    

def dense_core_shared(inputs_core, shared_block = 1,non_shared_block = 1, units = 64, branches = 14):
    non_shared_branch = []
    for branch in range(branches):
        x = inputs_core
        for block in range(non_shared_block):
            x = Dense(units,activation ='selu',kernel_initializer=tf.keras.initializers.RandomUniform(seed=7))(x)
            # x = BatchNormalization()(x)
        non_shared_branch.append(tf.expand_dims(x,2))
    x = Concatenate(axis = 2)(non_shared_branch)
   
    for block in range(shared_block):
        x = SharedWeightsDenseLayer(input_dim = x.shape[1],units = units,shares = 14)(x)        
        # x = BatchNormalization(axis = 1)(x)
    output_layer = SharedWeightsDenseLayer(input_dim = x.shape[1],units = 256,activation = False,shares = 14)(x)   
    return output_layer 


#### Training high level function
def train_model(training_type,byte,convolution_blocks , kernel_size,filters, strides , pooling_size,dense_blocks,dense_units,mlp,target):
    epochs = 50
    batch_size = 250
    n_traces = 5000
    
    if 'single_task' in training_type:
        X_profiling , validation_data = load_dataset(target,byte,n_traces = n_traces,dataset = 'training')
        model_t = '{}_{}'.format('model',training_type) 
    elif 'multi_task' in training_type:
        X_profiling , validation_data = load_dataset_multi('k1',n_traces = n_traces,dataset = 'training') 
        model_t = 'model_multi_task'
        
    elif 'target' in training_type:
        X_profiling , validation_data = load_dataset_multi('k1',n_traces = n_traces,dataset = 'training') 
        model_t = 'model_target'    
        
        
    else:
        print('OOPS')
    
    window =  X_profiling.element_spec[0]['traces'].shape[0]
    monitor = 'val_accuracy'
    mode = 'max'

    if model_t == 'model_single_task':
        model = model_single_task(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,mlp = mlp,input_length = window)     
        id_model = 'cb{}ks{}f{}s{}ps{}db{}du{}'.format(convolution_blocks , kernel_size,filters, strides , pooling_size,dense_blocks,dense_units)
        file_name = '{}_{}_{}_{}'.format(model_t,target,byte,id_model) 
    elif model_t == 'model_multi_task':

        model = model_multi_task(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,input_length = window)         
        monitor = 'val_loss'   
        mode = 'min'       
        id_model = 'cb{}ks{}f{}s{}ps{}db{}du{}'.format(convolution_blocks , kernel_size,filters, strides , pooling_size,dense_blocks,dense_units)
        file_name = '{}_all_{}'.format(model_t,id_model) 
    elif model_t == 'model_target':
        
        model = model_target(input_length = window)                  
        monitor = 'val_loss'   
        mode = 'min'           
        file_name = '{}'.format(model_t)  
    else:
        print('Some error here')

    
    X_profiling = X_profiling.shuffle(len(X_profiling)).batch(batch_size) 
    validation_data = validation_data.batch(batch_size)

    callbacks = tf.keras.callbacks.ModelCheckpoint(
                                filepath= MODEL_FOLDER+ file_name+'.h5'
                                , save_best_only=False, save_weights_only=True, mode='auto', save_freq=1)

    

    
    history = model.fit(X_profiling, batch_size=batch_size, verbose = 1, epochs=epochs, validation_data=validation_data,callbacks =callbacks)
    print('Saved model {} ! '.format(file_name))
 
    file = open(METRICS_FOLDER+'history_training_'+(file_name ),'wb')
    pickle.dump(history.history,file)
    file.close()

    
    
    return model






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--TARGET', action="store_true", dest="TARGET",
                        help='Classical training of the intermediates', default=False)
    parser.add_argument('--SINGLE',   action="store_true", dest="SINGLE", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI',   action="store_true", dest="MULTI", help='Adding the masks to the labels', default=False)
    parser.add_argument('--ALL',   action="store_true", dest="ALL", help='Adding the masks to the labels', default=False)
    args            = parser.parse_args()
  

    TARGET        = args.TARGET
    SINGLE        = args.SINGLE
    MULTI = args.MULTI

    ALL = args.ALL

    TARGETS = {}
    if SINGLE:
        targets = ['s1^r','t1^r','t1^ri','t1^i','i','r']
        
        for model_random in tqdm(range(100)):
            convolution_blocks = 1
            kernel_size = sorted(np.random.randint(16,64,size = convolution_blocks))       
            filters = np.random.randint(3,16)
            strides = np.random.randint(10,30)
            pooling_size = np.random.randint(2,5)
            dense_blocks = np.random.randint(1,5)
            dense_units = np.random.randint(8,512)    
            
            if model_random % 2 == 0:
                mlp = True
            else :
                mlp = False
                   
    
            for target in targets:
                for byte in range(2,16):
                    process_eval = Process(target=train_model, args=('single_task',byte,convolution_blocks , kernel_size,filters, strides , pooling_size,dense_blocks,dense_units,mlp,target))
                    process_eval.start()
                    process_eval.join()    
    elif MULTI:
        for model_random in tqdm(range(1000)):
            convolution_blocks = 1
            kernel_size = sorted(np.random.randint(16,64,size = convolution_blocks))       
            filters = np.random.randint(3,16)
            strides = np.random.randint(10,30)
            pooling_size = np.random.randint(2,5)
            dense_blocks = np.random.randint(1,5)
            dense_units = np.random.randint(64,512)    

            process_eval = Process(target=train_model, args=('multi_task',byte,convolution_blocks , kernel_size,filters, strides , pooling_size,dense_blocks,dense_units,mlp,None))
            process_eval.start()
            process_eval.join()    
    elif TARGET:
        process_eval = Process(target=train_model, args=('target','all',None , None,None, None , None,None,None,None,None))
        process_eval.start()
        process_eval.join()      
    else:
        print('No training mode selected')

                                


    print("$ Done !")
            
        
        
