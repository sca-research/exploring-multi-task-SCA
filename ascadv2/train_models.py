import argparse
import os
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Softmax, Dense, Input,Conv1D, AveragePooling1D, BatchNormalization ,Concatenate  , Add 
from tensorflow.keras.optimizers import Adam

from multiprocessing import Process


# import dataset paths and variables
from utility import  METRICS_FOLDER , MODEL_FOLDER

# import custom layers
from utility import MultiLayer , XorLayer , SharedWeightsDenseLayer  , InvSboxLayer

from utility import load_dataset, load_dataset_multi , load_dataset_hierarchical


import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()


seed = 42


tf.random.set_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

###########################################################################

def model_multi(learning_rate=0.001, classes=256,shared = False , name ='',summary = True,seed = 42):
    
    inputs_dict = {}
    outputs = {}
    
    inputs_block  = Input(shape = (93,16) ,name = 'inputs_block')
    inputs_dict['inputs_block'] = inputs_block 

    inputs_alpha  = Input(shape = (2000,1) ,name = 'inputs_alpha')
    inputs_dict['inputs_alpha'] = inputs_alpha 
    alpha_core = cnn_core(inputs_alpha,1,[32],16,10,5,seed = seed)
    alpha_core = dense_core(alpha_core,1,64,activated = True,name = 'alpha',seed = seed)
    # outputs['output_alpha'] = alpha_core
    
    inputs_rin  = Input(shape = (1000,1) ,name = 'inputs_rin')
    inputs_dict['inputs_rin'] = inputs_rin  
    rin_core = cnn_core(inputs_rin,1,[32],16,5,5,seed = seed)
    rin_core = dense_core(rin_core,1,64,activated = True,name = 'rin',seed = seed)
    # outputs['output_rin'] = rin_core
    
    inputs_beta  = Input(shape = (200,1) ,name = 'inputs_beta')    
    inputs_dict['inputs_beta'] = inputs_beta  
    beta_core = cnn_core(inputs_beta,1,[32],16,1,5,seed = seed)
    beta_core = dense_core(beta_core,1,64,activated = True,name = 'beta',seed = seed)
    # outputs['output_beta'] = beta_core


    metrics = {}
    s_branch = []
    t_branch = []
    branches = []
    for byte in range(16):
        block_core = SharedWeightsDenseLayer(input_dim = inputs_block.shape[1],units = 64,shares = 16,seed = seed)(inputs_block)      
        block_core = BatchNormalization(axis = 1)(block_core)
        block_core = Flatten()(block_core)
        branches.append(tf.expand_dims(block_core,2))
    block_core = Concatenate(axis = 2)(branches)

    if shared:
        s_branch = dense_core_shared(block_core,shared_block = 1,non_shared_block = 1,units = 8,output_units=256,split = True,seed = seed)
        t_branch = dense_core_shared(block_core,shared_block = 1,non_shared_block = 1,units = 8,output_units=256,split = True,seed = seed)

    else:
        for byte in range(16):   
            s_beta_core = dense_core(block_core[:,:,byte],2,8,seed = seed)
            s_branch.append(tf.expand_dims(s_beta_core,2))

            t_rin_core = dense_core(block_core[:,:,byte],2,8,seed = seed)
            t_branch.append(tf.expand_dims(t_rin_core,2))
        s_branch = Concatenate(axis = 2)(s_branch)
        t_branch = Concatenate(axis = 2)(t_branch)

    for byte in range(16):     

        xor_sj_beta =  XorLayer(name = 'xor_sj_beta_{}'.format(byte))([s_branch[:,:,byte],beta_core])
        sj = MultiLayer(name = 'multi_s_{}'.format(byte))([xor_sj_beta,alpha_core])
        outputs['output_sj_{}'.format(byte)] = Softmax(name = 'output_sj_{}'.format(byte))(sj)     

        xor_tj_rin =  XorLayer(name = 'xor_tj_rin_{}'.format(byte))([t_branch[:,:,byte],rin_core])
        tj = MultiLayer(name = 'multi_t_{}'.format(byte))([xor_tj_rin,alpha_core])

        outputs['output_tj_{}'.format(byte)] = Softmax(name = 'output_tj_{}'.format(byte))(tj)     
        metrics['output_tj_{}'.format(byte)] ='accuracy'


          
    losses = {}   

    # 
    for k , v in outputs.items():
        if not 'sig' in k:
            losses[k] = 'categorical_crossentropy'
        else:
            losses[k] = 'binary_crossentropy'

    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')    

    if summary:
        model.summary()
    return model  , losses  ,metrics 




def model_single_task(s = False, t = False,seed = 42,alpha_known = True, summary = True):
    inputs_dict = {}
    outputs = {}
    
    inputs_block  = Input(shape = (93,16) ,name = 'inputs_block')
    inputs_dict['inputs_block'] = inputs_block 

    if alpha_known:
        alpha = Input(shape = (256,))
        inputs_dict['alpha'] = alpha       
    else:
        
        inputs_alpha  = Input(shape = (2000,1) ,name = 'inputs_alpha')
        inputs_dict['inputs_alpha'] = inputs_alpha 
        alpha_core = cnn_core(inputs_alpha,1,[32],16,10,5,seed = seed)
        alpha = dense_core(alpha_core,1,64,activated = True,name = 'alpha',seed = seed)
    # outputs['output_alpha'] = alpha_core
    

    if s:
        inputs_beta  = Input(shape = (200,1) ,name = 'inputs_beta')    
        inputs_dict['inputs_beta'] = inputs_beta  
        beta_core = cnn_core(inputs_beta,1,[32],16,1,5,seed = seed)
        mask_core = dense_core(beta_core,1,64,activated = True,name = 'beta',seed = seed)
    if t:
  
    
        inputs_rin  = Input(shape = (1000,1) ,name = 'inputs_rin')
        inputs_dict['inputs_rin'] = inputs_rin  
        rin_core = cnn_core(inputs_rin,1,[32],16,5,5,seed = seed)
        mask_core = dense_core(rin_core,1,64,activated = True,name = 'rin',seed = seed)
        # outputs['output_rin'] = rin_core
        

    metrics = {}

    block_core = SharedWeightsDenseLayer(input_dim = inputs_block.shape[1],units = 64,shares = 16,seed = seed)(inputs_block)      
    block_core = BatchNormalization(axis = 1)(block_core)
    block_core = Flatten()(block_core)
    intermediate_core = dense_core(block_core,2,8,seed = seed,activated = True)
           
     
    xor =  XorLayer(name = 'xor')([intermediate_core,mask_core])
    mult = MultiLayer(name = 'output')([xor,alpha])

    outputs['output'] = mult     
    metrics['output'] ='accuracy'
    losses = {}
    losses['output'] = 'categorical_crossentropy'
    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_single')    
    return model  , losses  ,metrics 

def model_multi_task_s_only(seed = 42,shared = False,known_alpha = False,summary = True):
    inputs_dict = {}
    outputs = {}
    
    inputs_block  = Input(shape = (93,16) ,name = 'inputs_block')
    inputs_dict['inputs_block'] = inputs_block 

    if known_alpha:
        alpha = Input(shape = (256,))
        inputs_dict['alpha'] = alpha
    else:
        inputs_alpha  = Input(shape = (2000,1) ,name = 'inputs_alpha')
        inputs_dict['inputs_alpha'] = inputs_alpha 
        alpha_core = cnn_core(inputs_alpha,1,[32],16,10,5,seed = seed)
        alpha = dense_core(alpha_core,1,64,activated = True,name = 'alpha',seed = seed)
        outputs['output_alpha'] = alpha
    

    
    inputs_beta  = Input(shape = (200,1) ,name = 'inputs_beta')    
    inputs_dict['inputs_beta'] = inputs_beta  
    beta_core = cnn_core(inputs_beta,1,[32],16,1,5,seed = seed)
    beta_core = dense_core(beta_core,1,64,activated = True,name = 'beta',seed = seed)


    metrics = {}
    s_branch = []

    branches = []
    for byte in range(16):
        block_core = SharedWeightsDenseLayer(input_dim = inputs_block.shape[1],units = 64,shares = 16,seed = seed)(inputs_block)      
        block_core = BatchNormalization(axis = 1)(block_core)
        block_core = Flatten()(block_core)
        branches.append(tf.expand_dims(block_core,2))
    block_core = Concatenate(axis = 2)(branches)

    if shared:
        s_branch = dense_core_shared(block_core,shared_block = 1,non_shared_block = 1,units = 8,output_units=256,split = True,seed = seed)
    else:
        for byte in range(16):   
            s_beta_core = dense_core(block_core[:,:,byte],2,8,seed = seed)
            s_branch.append(tf.expand_dims(s_beta_core,2))

        s_branch = Concatenate(axis = 2)(s_branch)

    for byte in range(16):   
        xor_sj_beta =  XorLayer(name = 'xor_sj_beta_{}'.format(byte))([s_branch[:,:,byte],beta_core])
        sj = MultiLayer(name = 'multi_s_{}'.format(byte))([xor_sj_beta,alpha])
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_sj_{}'.format(byte))(sj)     
        metrics['output_{}'.format(byte)] ='accuracy'

          
    losses = {}   

    for k , v in outputs.items():
    
        losses[k] = 'categorical_crossentropy'
    

   
    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')    
    if summary:
        model.summary()
    return model  , losses  ,metrics 


def model_multi_task_t_only(shared = False,seed = 42,known_alpha = False,summary = True):
    inputs_dict = {}
    outputs = {}
    
    inputs_block  = Input(shape = (93,16) ,name = 'inputs_block')
    inputs_dict['inputs_block'] = inputs_block 


    if known_alpha:
        alpha = Input(shape = (256,))
        inputs_dict['alpha'] = alpha
    else:
        inputs_alpha  = Input(shape = (2000,1) ,name = 'inputs_alpha')
        inputs_dict['inputs_alpha'] = inputs_alpha 
        alpha_core = cnn_core(inputs_alpha,1,[32],16,10,5,seed = seed)
        alpha = dense_core(alpha_core,1,64,activated = True,name = 'alpha',seed = seed)
        outputs['output_alpha'] = alpha
    
    inputs_rin  = Input(shape = (1000,1) ,name = 'inputs_rin')
    inputs_dict['inputs_rin'] = inputs_rin  
    rin_core = cnn_core(inputs_rin,1,[32],16,5,5,seed = seed)
    rin_core = dense_core(rin_core,1,64,activated = True,name = 'rin',seed = seed)

    # outputs['output_rin'] = rin_core
    

    metrics = {}

    t_branch = []
    branches = []
    for byte in range(16):
        block_core = SharedWeightsDenseLayer(input_dim = inputs_block.shape[1],units = 64,shares = 16,seed = seed)(inputs_block)      
        block_core = BatchNormalization(axis = 1)(block_core)
        block_core = Flatten()(block_core)
        branches.append(tf.expand_dims(block_core,2))
    block_core = Concatenate(axis = 2)(branches)

    if shared:
        t_branch = dense_core_shared(block_core,shared_block = 1,non_shared_block = 1,units = 8,output_units=256,split = True,seed = seed)

    else:
        for byte in range(16):   

            t_rin_core = dense_core(block_core[:,:,byte],2,8,seed = seed)
            t_branch.append(tf.expand_dims(t_rin_core,2))
     
        t_branch = Concatenate(axis = 2)(t_branch)

    for byte in range(16):     
        xor_tj_rin =  XorLayer(name = 'xor_tj_rin_{}'.format(byte))([t_branch[:,:,byte],rin_core])
        tj = MultiLayer(name = 'multi_t_{}'.format(byte))([xor_tj_rin,alpha])

        outputs['output_{}'.format(byte)] = Softmax(name = 'output_tj_{}'.format(byte))(tj)     
        metrics['output_{}'.format(byte)] ='accuracy'


          
    losses = {}   

    for k , v in outputs.items():
    
        losses[k] = 'categorical_crossentropy'
    

    
    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')    

    if summary:
        model.summary()
    return model  , losses  ,metrics 






######################## ARCHITECTURE BUILDING ################################

def cnn_core(inputs_core,convolution_blocks , kernel_size,filters, strides , pooling_size,seed = 42):
    x = inputs_core
    for block in range(convolution_blocks):
        x = Conv1D(kernel_size=(kernel_size[block],), strides=strides, filters=filters, activation='selu', padding='same',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)    
        x = BatchNormalization()(x)
        x = AveragePooling1D(pool_size = pooling_size)(x)
    
    output_layer = Flatten()(x) 

    return output_layer


def dense_core(inputs_core,dense_blocks,dense_units,activated = False,name = '',seed = 42):
    x = inputs_core
    
    for block in range(dense_blocks):
        x = Dense(dense_units, activation='selu',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)        
        x = BatchNormalization()(x)
        
    if activated:
        output_layer = Dense(256,activation ='softmax' ,name = 'output_{}'.format(name),kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)  
    else:
        output_layer = Dense(256,kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)   
    return output_layer    

def dense_core_shared(inputs_core, shared_block = 1,non_shared_block = 1, units = 8, branches = 16,output_units = 32,precision = 'float32',split = False,seed = 42):
    non_shared_branch = []
    if non_shared_block > 0:
        for branch in range(branches):
            x = inputs_core
         
            x = Dense(units,activation ='selu',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x[:,:,branch] if split else x)
            x = BatchNormalization()(x)
            non_shared_branch.append(tf.expand_dims(x,2))
        x = Concatenate(axis = 2)(non_shared_branch)
    else:
        x = inputs_core

    for block in range(shared_block):
        x = SharedWeightsDenseLayer(input_dim = x.shape[1],units = units,shares = branches,seed = seed)(x)        
        x = BatchNormalization(axis = 1)(x)
    output_layer = SharedWeightsDenseLayer(input_dim = x.shape[1],units = output_units,activation = False,shares = branches,precision = precision,seed = seed)(x)   
    return output_layer 

#### Training high level function

def train_model(shared,training_type,byte):

    batch_size = 500
    
    known_alpha = 'first' in training_type
    n_traces = 225000 if known_alpha else 450000

    
    model_t = 'model_{}_{}{}'.format(training_type, 'shared' if shared else 'nshared','_'+str(byte) if not (byte is None) else '')
    
    if training_type == 'multi':     
        model , losses , metrics  = model_multi(shared = shared,seed = seed)   
    elif training_type == 'multi_t':
        model , losses , metrics  = model_multi_task_t_only(shared = shared,seed = seed, known_alpha = known_alpha)
    elif training_type == 'multi_s':       
        model , losses , metrics  = model_multi_task_s_only(shared = shared,seed = seed, known_alpha = known_alpha)
    elif 'single' in training_type:
        model , losses , metrics  = model_single_task(s = 'single_s' in training_type, t = 'single_t' in training_type,seed = seed, alpha_known = 'first' in training_type)
    else:
        print('')



    learning_rates = [0.001,0.0001,0.00001]
    epochs  = [25,4,1]
    callbacks = tf.keras.callbacks.ModelCheckpoint(
                                    filepath= MODEL_FOLDER+ model_t+'.h5',
                                    save_weights_only=True,
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=True)
    for cycle in range(3):

        optimizer = Adam(learning_rate=learning_rates[cycle])
        model.compile(loss=losses, optimizer=optimizer, metrics=metrics)
        if 'multi' in training_type:           
            X_profiling , validation_data = load_dataset_multi(n_traces = n_traces,only_t = training_type == 'multi_t',only_s = training_type == 'multi_s',dataset = 'training',known_alpha = known_alpha) 
        else:
            X_profiling , validation_data = load_dataset(byte,n_traces = n_traces,t = 'single_t' in training_type , alpha_known = 'first' in training_type,dataset = 'training') 
        X_profiling = X_profiling.shuffle(len(X_profiling)).batch(batch_size) 
        validation_data = validation_data.batch(batch_size)
        history = model.fit(X_profiling, batch_size=batch_size, verbose = 1, epochs=epochs[cycle], validation_data=validation_data,callbacks =callbacks)
    print('Saved model ! ')    
    return model






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--SHARED',   action="store_true", dest="SHARED", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI',   action="store_true", dest="MULTI", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI_S',   action="store_true", dest="MULTI_S", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI_T',   action="store_true", dest="MULTI_T", help='Adding the masks to the labels', default=False)
    parser.add_argument('--SINGLE_S',   action="store_true", dest="SINGLE_S", help='Adding the masks to the labels', default=False)
    parser.add_argument('--SINGLE_T',   action="store_true", dest="SINGLE_T", help='Adding the masks to the labels', default=False)
    parser.add_argument('--FIRST',   action="store_true", dest="FIRST", help='Adding the masks to the labels', default=False)

    args            = parser.parse_args()
  
    SHARED        = args.SHARED
    MULTI = args.MULTI
    MULTI_S = args.MULTI_S
    MULTI_T= args.MULTI_T
    SINGLE_T=  args.SINGLE_T
    SINGLE_S = args.SINGLE_S
    FIRST = args.FIRST
    

    if MULTI:
        TRAINING_TYPE = 'multi'
    elif MULTI_T:
        TRAINING_TYPE = 'multi_t'
        if FIRST:
            TRAINING_TYPE += '_first'
    elif MULTI_S:
        TRAINING_TYPE = 'multi_s'
        if FIRST:
            TRAINING_TYPE += '_first'
    elif SINGLE_T:
        TRAINING_TYPE = 'single_t'
        if FIRST:
            TRAINING_TYPE += '_first'
    elif SINGLE_S:
        TRAINING_TYPE = 'single_s'
        if FIRST:
            TRAINING_TYPE += '_first'
    else:
        print('')
    TARGETS = {}
    seeds_random = np.random.randint(0,9999,size = 10)
    for seed in seeds_random:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        if not 'single' in TRAINING_TYPE:

            process_eval = Process(target=train_model, args=(SHARED,TRAINING_TYPE,'all'))
            process_eval.start()
            process_eval.join() 
        else:
   
            for byte in range(16):
                process_eval = Process(target=train_model, args=(False,TRAINING_TYPE + ('' if not FIRST else '_first'),byte))
                process_eval.start()
                process_eval.join()                                    


    print("$ Done !")
            
        
        
