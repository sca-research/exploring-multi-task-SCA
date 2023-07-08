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
        
        self.loss_tracker = tf.keras.metrics.Mean(name= 'loss')        


    
    def train_step(self, data):
        x, y = data
   
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(
                y,
                y_pred,
                regularization_losses=self.losses,
            )
      
       
           
            sum_losses = loss / 14
            for byte in range(2,16):
                loss = loss + tf.math.pow(tf.keras.losses.categorical_crossentropy(y['output_{}'.format(byte)],y_pred['output_{}'.format(byte)]) - sum_losses,2) / 14
           
            
                
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
         
       
        sum_losses = loss / 14
        for byte in range(2,16):
            loss = loss + tf.math.pow(tf.keras.losses.categorical_crossentropy(y['output_{}'.format(byte)],y_pred['output_{}'.format(byte)]) - sum_losses,2) / 14
       
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

def model_single_task(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = True):
    inputs_dict = {}
    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    branch = input_layer_creation(inputs,input_length)
    inputs_dict['traces'] = inputs   
    
    branch = input_layer_creation(inputs,input_length)
    branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size)

    outputs = {}
      
    mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True)
    intermediate_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = False)
    
    xor = XorLayer(name ='xor' )([mask_branch,intermediate_branch]) 
    outputs['output'] = Softmax(name ='output')(xor)
    
    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_single_task_xor_{}'.format(name))

    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if summary:
        model.summary() 
    return model




def model_multi_task_single_target_one_shared_mask(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,multi_model = False,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length)
    branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size)

    mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True)


    outputs = {} 

    for byte in range(2,16):
        
        intermediate_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = False)
        
        xor = XorLayer(name ='xor_{}'.format(byte) )([intermediate_branch,mask_branch])        
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_t_{}'.format(byte))(xor)
        
    losses = {}   
    




    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')      if not multi_model else Multi_Model(inputs = inputs_dict,outputs = outputs)  
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'])
    
    if summary:
        model.summary()
    return model   


def model_multi_task_single_target_one_shared_mask_shared_branch(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,multi_model = False,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length)
    branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size)

    mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True)
    shared_branch_t_i = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14)

    outputs = {} 

    for byte in range(2,16):
        
        xor = XorLayer(name ='xor_{}'.format(byte) )([shared_branch_t_i[:,:,byte-2],mask_branch])        
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_{}'.format(byte))(xor)
        
    losses = {}   
    



    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')      if not multi_model else Multi_Model(inputs = inputs_dict,outputs = outputs)  
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy']) 
    if summary:
        model.summary()
    return model   


def model_multi_task_single_target(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,multi_model = False,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length)
    branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size)

    shared_branch_s_r = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14)
    shared_branch_r = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14)
    
    outputs = {} 
    
    for byte in range(2,16):
        
        shared_branch_r_activated = Softmax()(shared_branch_r[:,:,byte-2])   
        
        xor = XorLayer(name ='xor_{}'.format(byte) )([shared_branch_s_r[:,:,byte-2],shared_branch_r_activated])        
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_{}'.format(byte))(xor)
        
    losses = {}   
    



    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')      if not multi_model else Multi_Model(inputs = inputs_dict,outputs = outputs)  
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy']) 
    if summary:
        model.summary()
    return model   



def model_multi_task_single_target_not_shared(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,multi_model = False,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   
   
 
    branch = input_layer_creation(inputs,input_length)
    branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size)


    outputs = {} 
    
    for byte in range(2,16):
        
        intermediate_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = False)
        mask_branch = dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True)
                
        xor = XorLayer(name ='xor_{}'.format(byte) )([intermediate_branch,mask_branch])        
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_{}'.format(byte))(xor)
    losses = {}   
    



    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')      if not multi_model else Multi_Model(inputs = inputs_dict,outputs = outputs)  
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy']) 
    if summary:
        model.summary()
    return model   




def model_multi_task_multi_target(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,multi_model = False,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = False):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    
    inputs_dict['traces'] = inputs   
   
   
 
    branch = input_layer_creation(inputs,input_length)
    branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size)

    shared_branch_t_r = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14)
    shared_branch_s_r = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14)
    shared_branch_r = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14)
    xor_operation = XorLayer(name = 'xor_layer')

    outputs = {} 

    for byte in range(2,16):
        
        shared_branch_r_activated = Softmax()(shared_branch_r[:,:,byte-2])        
        
        xor_s = xor_operation([shared_branch_s_r[:,:,byte-2],shared_branch_r_activated])        
        inv_sbox = Softmax(name = 'output_s_{}'.format(byte))(xor_s)
        outputs['output_s_{}'.format(byte)] = inv_sbox
        
        inv_sbox = InvSboxLayer(name = 'inv_sbox_{}'.format(byte))(inv_sbox)
        xor_t_r = xor_operation([shared_branch_t_r[:,:,byte-2],shared_branch_r_activated])        
   
        
        add_shares = Multiply()([xor_t_r,inv_sbox])

        outputs['output_{}'.format(byte)] = Softmax(name = 'output_t_{}'.format(byte))(add_shares)
        
    losses = {}   
    


    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')      if not multi_model else Multi_Model(inputs = inputs_dict,outputs = outputs)  
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy']) 
    if summary:
        model.summary()
    return model   


def model_multi_task_single_target_multi_shares(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,multi_model = False,input_length=1000, learning_rate=0.001, classes=256 , name ='',summary = True):
    
    inputs_dict = {}    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
 
    inputs_dict['traces'] = inputs   
   
   
 
    branch = input_layer_creation(inputs,input_length)
    branch = cnn_core(branch,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = strides, pooling_size = pooling_size)

    branch_i= dense_core(branch,dense_blocks = dense_blocks,dense_units = dense_units,batch_norm = False,activated = True)
    

    shared_branch_t_r = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14)
    shared_branch_t_i = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14)
    shared_branch_t_ri = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14)
    shared_branch_r = dense_core_shared(branch,shared_block = 1, non_shared_block = 1, units = 200,branches = 14)
    
    xor_operation = XorLayer(name = 'xor_layer')

    outputs = {} 

    for byte in range(2,16):
        
        shared_branch_r_activated = Softmax()(shared_branch_r[:,:,byte-2])     

        xor_t_r = xor_operation([shared_branch_t_r[:,:,byte-2],shared_branch_r_activated])        
        xor_t_i = xor_operation([shared_branch_t_i[:,:,byte-2],branch_i])
        xor_r_i = xor_operation([shared_branch_r[:,:,byte-2],branch_i])
        xor_t_ri = xor_operation([shared_branch_t_ri[:,:,byte-2],xor_r_i])        
        
        add_shares = Add()([xor_t_r,xor_t_i,xor_t_ri])

        outputs['output_{}'.format(byte)] = Softmax(name = 'output_{}'.format(byte))(add_shares)
        



        
    losses = {}   
    


    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')      if not multi_model else Multi_Model(inputs = inputs_dict,outputs = outputs)  
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy']) 
    if summary:
        model.summary()
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
            #x = BatchNormalization()(x)
        non_shared_branch.append(tf.expand_dims(x,2))
    x = Concatenate(axis = 2)(non_shared_branch)
   
    for block in range(shared_block):
        x = SharedWeightsDenseLayer(input_dim = x.shape[1],units = units,shares = 14)(x)        
        #x = BatchNormalization(axis = 1)(x)
    output_layer = SharedWeightsDenseLayer(input_dim = x.shape[1],units = 256,activation = False,shares = 14)(x)   
    return output_layer 


#### Training high level function
def train_model(training_type,byte,convolution_blocks , kernel_size,filters, strides , pooling_size,dense_blocks,dense_units,multi_model):
    epochs = 100 
    batch_size = 100
    n_traces = 50000
    
    if 'single_task' in training_type:
        X_profiling , validation_data = load_dataset(byte,n_traces = n_traces,dataset = 'training')
        model_t = '{}_{}'.format('multi_model' if multi_model else 'model',training_type) 
    elif ('multi_task_single_target_one_shared_mask' in training_type) or ('multi_task_single_target_multi_shares' in training_type) :
        X_profiling , validation_data = load_dataset_multi('t1',n_traces = n_traces,dataset = 'training') 
        model_t = '{}_{}'.format('multi_model' if multi_model else 'model',training_type)     
        
    elif 'multi_task_single_target' in training_type:
        X_profiling , validation_data = load_dataset_multi('s1',n_traces = n_traces,dataset = 'training') 
        model_t = '{}_{}'.format('multi_model' if multi_model else 'model',training_type)
        
    elif 'multi_task_multi_target' in training_type:
        X_profiling , validation_data = load_dataset_multi('k1',n_traces = n_traces,dataset = 'training') 
        model_t = '{}_{}'.format('multi_model' if multi_model else 'model',training_type)       
        
        
    else:
        X_profiling , validation_data = load_dataset_multi('t1',n_traces = n_traces,dataset = 'training') 
        model_t = 'model_multi_task'
    
    window =  X_profiling.element_spec[0]['traces'].shape[0]
    monitor = 'val_accuracy'
    mode = 'max'

    if model_t == 'model_single_task':
        model = model_single_task(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,multi_model = multi_model,input_length = window)         
    elif model_t == '{}_multi_task_single_target'.format('multi_model' if multi_model else 'model'):

        model = model_multi_task_single_target(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,multi_model = multi_model,input_length = window)         
        monitor = 'val_loss'   
        mode = 'min'     
    elif model_t == '{}_multi_task_single_target_not_shared'.format('multi_model' if multi_model else 'model'):

        model = model_multi_task_single_target_not_shared(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,multi_model = multi_model,input_length = window)         
        monitor = 'val_loss'   
        mode = 'min'     

    elif model_t == '{}_multi_task_single_target_one_shared_mask'.format('multi_model' if multi_model else 'model'):
        
        model = model_multi_task_single_target_one_shared_mask(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,multi_model = multi_model,input_length = window)                  
        monitor = 'val_loss'   
        mode = 'min'
        
        
    elif model_t == '{}_multi_task_single_target_one_shared_mask_shared_branch'.format('multi_model' if multi_model else 'model'):
        
        model = model_multi_task_single_target_one_shared_mask_shared_branch(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,multi_model = multi_model,input_length = window)                  
        monitor = 'val_loss'   
        mode = 'min'        
    elif model_t == '{}_multi_task_multi_target'.format('multi_model' if multi_model else 'model'):
        
        model = model_multi_task_multi_target(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,multi_model = multi_model,input_length = window)                  
        monitor = 'val_loss'   
        mode = 'min'           
    elif model_t == '{}_multi_task_single_target_multi_shares'.format('multi_model' if multi_model else 'model'):
        
        model = model_multi_task_single_target_multi_shares(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,multi_model = multi_model,input_length = window)                  
        monitor = 'val_loss'   
        mode = 'min'        
    else:
        print('Some error here')

    
    X_profiling = X_profiling.shuffle(len(X_profiling)).batch(batch_size) 
    validation_data = validation_data.batch(batch_size)
    id_model = 'cb{}ks{}f{}s{}ps{}db{}du{}'.format(convolution_blocks , kernel_size,filters, strides , pooling_size,dense_blocks,dense_units)
    file_name = '{}_all_{}'.format(model_t,id_model) 
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
    parser.add_argument('--MULTI_MODEL', action="store_true", dest="MULTI_MODEL",
                        help='Classical training of the intermediates', default=False)
    parser.add_argument('--SINGLE',   action="store_true", dest="SINGLE", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI',   action="store_true", dest="MULTI", help='Adding the masks to the labels', default=False)
    parser.add_argument('--ALL',   action="store_true", dest="ALL", help='Adding the masks to the labels', default=False)
    args            = parser.parse_args()
  

    MULTI_MODEL        = args.MULTI_MODEL
    SINGLE        = args.SINGLE
    MULTI = args.MULTI

    ALL = args.ALL

    TARGETS = {}
    if SINGLE:
        training_types = ['multi_task_single_target_one_shared_mask','multi_task_single_target_one_shared_mask_shared_branch']

    elif MULTI:
        training_types = ['multi_task_single_target_multi_shares']
    elif ALL:
        training_types = ['single_task_xor', 'single_task_twin' , 'multi_task']
    else:
        print('No training mode selected')

    for model_random in tqdm(range(1)):

        # convolution_blocks = np.random.randint(1,3)
        # kernel_size = sorted(np.random.randint(16,64,size = convolution_blocks))       
        # filters = np.random.randint(3,16)
        # strides = np.random.randint(2,30)
        # pooling_size = np.random.randint(2,5)
        # dense_blocks = np.random.randint(1,5)
        # dense_units = (np.random.randint(64,512)//14) * 14 
 
        convolution_blocks = 1
        kernel_size = [34]    
        filters = 4
        strides = 17
        pooling_size = 2
        dense_blocks = 2
        dense_units = 200      

        for training_type in training_types:
            process_eval = Process(target=train_model, args=(training_type,'all',convolution_blocks , kernel_size,filters, strides , pooling_size,dense_blocks,dense_units,MULTI_MODEL))
            process_eval.start()
            process_eval.join()                                    


    print("$ Done !")
            
        
        
