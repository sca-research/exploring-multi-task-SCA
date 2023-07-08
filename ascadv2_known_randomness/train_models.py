import argparse
import os
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Softmax, Dense, Input,Conv1D, AveragePooling1D, BatchNormalization,Multiply ,Concatenate , Activation , Add , Reshape
from tensorflow.keras.optimizers import Adam

from multiprocessing import Process


# import dataset paths and variables
from utility import  METRICS_FOLDER , MODEL_FOLDER, VARIABLE_LIST

# import custom layers
from utility import MultiLayer , XorLayer , SharedWeightsDenseLayer  , InvSboxLayer

from utility import load_dataset, load_dataset_multi 





seed = 7


tf.random.set_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

###########################################################################






########################## FULLY EXTRACTED SCENARIO #################################################

def model_sbox_output( learning_rate=0.001, classes=256 , name ='',summary = True):
    
    inputs_dict = {}
    outputs = {}
    
    inputs_block  = Input(shape = (93,16) ,name = 'inputs_block')
    inputs_dict['inputs_block'] = inputs_block 
    block_core = SharedWeightsDenseLayer(input_dim = inputs_block.shape[1],units = 64,shares = 16)(inputs_block)      
    block_core = BatchNormalization(axis = 1)(block_core)
    

    inputs_alpha  = Input(shape = (2000,1) ,name = 'inputs_alpha')
    inputs_dict['inputs_alpha'] = inputs_alpha 
    alpha_core = cnn_core(inputs_alpha,1,[32],16,10,5)
    alpha_core = dense_core(alpha_core,1,64,activated = True,name = 'alpha')
    outputs['alpha'] = alpha_core
    

    
    inputs_beta  = Input(shape = (200,1) ,name = 'inputs_beta')    
    inputs_dict['inputs_beta'] = inputs_beta  
    beta_core = cnn_core(inputs_beta,1,[32],16,1,5)
    beta_core = dense_core(beta_core,1,64,activated = True,name = 'beta')
    outputs['beta'] = beta_core
    


    s_beta_core = dense_core_shared(block_core,output_units = 256)

    metrics = {}
    for byte in range(16): 
        
        outputs['output_s_beta_{}'.format(byte)] = Softmax(name = 'output_s_beta_{}'.format(byte))(s_beta_core[:,:,byte])     
        metrics['output_s_beta_{}'.format(byte)] ='accuracy'

          
    losses = {}   

    weights = {}
    for k , v in outputs.items():
       
        losses[k] = 'categorical_crossentropy'
 
        weights[k] = 1 if not 'output' in k else 1
        


    model = Model(inputs = inputs_dict,outputs = outputs)    

    if summary:
        model.summary()
    return model  , losses  ,metrics  , weights
def model_sbox_input( learning_rate=0.001, classes=256 , name ='',summary = True):
    
    inputs_dict = {}
    outputs = {}
    
    inputs_block  = Input(shape = (93,16) ,name = 'inputs_block')
    inputs_dict['inputs_block'] = inputs_block 
    block_core = SharedWeightsDenseLayer(input_dim = inputs_block.shape[1],units = 64,shares = 16)(inputs_block)      
    block_core = BatchNormalization(axis = 1)(block_core)
    
    
    inputs_alpha  = Input(shape = (2000,1) ,name = 'inputs_alpha')
    inputs_dict['inputs_alpha'] = inputs_alpha 
    alpha_core = cnn_core(inputs_alpha,1,[32],16,10,5)
    alpha_core = dense_core(alpha_core,1,64,activated = True,name = 'alpha')
    outputs['alpha'] = alpha_core
    
    inputs_rin  = Input(shape = (1000,1) ,name = 'inputs_rin')
    inputs_dict['inputs_rin'] = inputs_rin  
    rin_core = cnn_core(inputs_rin,1,[32],16,5,5)
    rin_core = dense_core(rin_core,1,64,activated = True,name = 'rin')
    outputs['rin'] = rin_core
    

    t_rin_core = dense_core_shared(block_core,output_units = 256)

    metrics = {}
    
    for byte in range(16): 
        
        outputs['output_t_rin_{}'.format(byte)] = Softmax(name = 'output_t_rin_{}'.format(byte))(t_rin_core[:,:,byte])     
        metrics['output_t_rin_{}'.format(byte)] ='accuracy'


          
    losses = {}   

    weights = {}
    for k , v in outputs.items():
       
        losses[k] = 'categorical_crossentropy'
 
        weights[k] = 1 if not 'output' in k else 1
        


    model = Model(inputs = inputs_dict,outputs = outputs)    

    if summary:
        model.summary()
    return model  , losses  ,metrics  , weights






def model_intermediate_single(learning_rate=0.001, classes=256 , name ='',summary = True):
    inputs_dict = {}
    outputs = {}
    

    inputs_intermediate  = Input(shape = (93,16) ,name = 'inputs_intermediate')
    inputs_dict['inputs_block'] = inputs_intermediate  
    block_core = SharedWeightsDenseLayer(input_dim = inputs_intermediate.shape[1],units = 64,shares = 16)(inputs_intermediate)      
    block_core = BatchNormalization(axis = 1)(block_core)
    block_core = Flatten()(block_core)

    block_core = dense_core(block_core,2,8,bn = False,activated = True)   

    metrics = {}

    outputs['output'] =  block_core
    metrics['output'] = 'accuracy'

          
    losses = {}   

    weights = {}
    for k , v in outputs.items():
       
        losses[k] = 'categorical_crossentropy'
 
        weights[k] = 1 if not 'output' in k else 1
        


    model = Model(inputs = inputs_dict,outputs = outputs)    

    if summary:
        model.summary()
    return model  , losses  ,metrics  , weights

def model_alpha_single(learning_rate=0.001, classes=256 , name ='',summary = True):
    inputs_dict = {}
    outputs = {}
    metrics = {}
    

    inputs_alpha  = Input(shape = (2000,1) ,name = 'inputs_alpha')
    inputs_dict['inputs_alpha'] = inputs_alpha 
    alpha_core = cnn_core(inputs_alpha,1,[32],16,10,5)
    alpha_core = dense_core(alpha_core,1,64,activated = True,name = 'alpha')
    outputs['output'] = alpha_core
    metrics['output'] = 'accuracy'

          
    losses = {}   

    weights = {}
    for k , v in outputs.items():
       
        losses[k] = 'categorical_crossentropy'
 
        weights[k] = 1 if not 'output' in k else 1
     
        


    model = Model(inputs = inputs_dict,outputs = outputs)    

    if summary:
        model.summary()
    return model  , losses  ,metrics  , weights

def model_rin_single(learning_rate=0.001, classes=256 , name ='',summary = True):
    inputs_dict = {}
    outputs = {}
    metrics = {}
    

    inputs_rin  = Input(shape = (1000,1) ,name = 'inputs_rin')
    inputs_dict['inputs_rin'] = inputs_rin  
    rin_core = cnn_core(inputs_rin,1,[32],16,5,5)
    rin_core = dense_core(rin_core,1,64,activated = True,name = 'rin')
    outputs['output'] = rin_core
    metrics['output'] = 'accuracy'

          
    losses = {}   

    weights = {}
    for k , v in outputs.items():
       
        losses[k] = 'categorical_crossentropy'
 
        weights[k] = 1 if not 'output' in k else 1
        


    model = Model(inputs = inputs_dict,outputs = outputs)    

    if summary:
        model.summary()
    return model  , losses  ,metrics  , weights


def model_beta_single(learning_rate=0.001, classes=256 , name ='',summary = True):
    inputs_dict = {}
    outputs = {}
    metrics = {}
    

    inputs_beta  = Input(shape = (200,1) ,name = 'inputs_beta')    
    inputs_dict['inputs_beta'] = inputs_beta  
    beta_core = cnn_core(inputs_beta,1,[32],16,1,5)
    beta_core = dense_core(beta_core,1,64,activated = True,name = 'beta')
    outputs['output'] = beta_core
    metrics['output'] = 'accuracy'

          
    losses = {}   

    weights = {}
    for k , v in outputs.items():
       
        losses[k] = 'categorical_crossentropy'
 
        weights[k] = 1 if not 'output' in k else 1
        


    model = Model(inputs = inputs_dict,outputs = outputs)    

    if summary:
        model.summary()
    return model  , losses  ,metrics  , weights

def model_flat( learning_rate=0.001, classes=256 , name ='',summary = True):
    
    inputs_dict = {}
    outputs = {}
    
    inputs_block  = Input(shape = (93,16) ,name = 'inputs_block')
    inputs_dict['inputs_block'] = inputs_block 

    
    

    
    inputs_alpha  = Input(shape = (2000,1) ,name = 'inputs_alpha')
    inputs_dict['inputs_alpha'] = inputs_alpha 
    alpha_core = cnn_core(inputs_alpha,1,[32],16,10,5)
    alpha_core = dense_core(alpha_core,1,64,activated = True,name = 'alpha')
    outputs['alpha'] = alpha_core
    
    inputs_rin  = Input(shape = (1000,1) ,name = 'inputs_rin')
    inputs_dict['inputs_rin'] = inputs_rin  
    rin_core = cnn_core(inputs_rin,1,[32],16,5,5)
    rin_core = dense_core(rin_core,1,64,activated = False,name = 'rin')
    outputs['rin'] = Softmax(name= 'rin')(rin_core)
    
    inputs_beta  = Input(shape = (200,1) ,name = 'inputs_beta')    
    inputs_dict['inputs_beta'] = inputs_beta  
    beta_core = cnn_core(inputs_beta,1,[32],16,1,5)
    beta_core = dense_core(beta_core,1,64,activated = False,name = 'beta')
    outputs['beta'] = Softmax(name= 'beta')(beta_core)
    

    for byte in range(16):
        block_core = SharedWeightsDenseLayer(input_dim = inputs_block.shape[1],units = 64,shares = 16)(inputs_block)      
        block_core = BatchNormalization(axis = 1)(block_core)
        block_core = Flatten()(block_core)
        s_beta = dense_core(block_core,2,8,activated = False)
        t_rin = dense_core(block_core,2,8,activated = False)
        outputs['s_beta_{}'.format(byte)] = Softmax(name = 'output_s_beta_{}'.format(byte))(s_beta)
        outputs['t_rin_{}'.format(byte)] =  Softmax(name = 'output_t_rin_{}'.format(byte))(t_rin)

    




        
              


          
    losses = {}   

    weights = {}
    metrics = {}
    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'

        weights[k] = 1 if not 'output' in k else 1
        metrics[k] = 'accuracy'
        


    model = Model(inputs = inputs_dict,outputs = outputs)    

    if summary:
        model.summary()
    return model  , losses  ,metrics  , weights

def model_hierarchical( learning_rate=0.001, classes=256 , name ='',summary = True):

    inputs_dict = {}
    outputs = {}
    
    inputs_block  = Input(shape = (93,16) ,name = 'inputs_block')
    inputs_dict['inputs_block'] = inputs_block 

    inputs_alpha  = Input(shape = (2000,1) ,name = 'inputs_alpha')
    inputs_dict['inputs_alpha'] = inputs_alpha 
    alpha_core = cnn_core(inputs_alpha,1,[32],16,10,5)
    alpha_core = dense_core(alpha_core,1,64,activated = True,name = 'alpha')
    outputs['alpha'] = alpha_core
    
    inputs_rin  = Input(shape = (1000,1) ,name = 'inputs_rin')
    inputs_dict['inputs_rin'] = inputs_rin  
    rin_core = cnn_core(inputs_rin,1,[32],16,5,5)
    rin_core = dense_core(rin_core,1,64,activated = True,name = 'rin')
    outputs['rin'] = rin_core
    
    inputs_beta  = Input(shape = (200,1) ,name = 'inputs_beta')    
    inputs_dict['inputs_beta'] = inputs_beta  
    beta_core = cnn_core(inputs_beta,1,[32],16,1,5)
    beta_core = dense_core(beta_core,1,64,activated = True,name = 'beta')
    outputs['beta'] = beta_core
    

    metrics = {}

    for byte in range(16):
        block_core = SharedWeightsDenseLayer(input_dim = inputs_block.shape[1],units = 64,shares = 16)(inputs_block)      
        block_core = BatchNormalization(axis = 1)(block_core)
        block_core = Flatten()(block_core)
        s_beta = dense_core(block_core,2,8,activated = False)
        t_rin = dense_core(block_core,2,8,activated = False)
        outputs['s_beta_{}'.format(byte)] = Softmax(name = 'output_s_beta_{}'.format(byte))(s_beta)
        outputs['t_rin_{}'.format(byte)] =  Softmax(name = 'output_t_rin_{}'.format(byte))(t_rin)

        xor_sj_beta =  XorLayer(name = 'xor_sj_beta_{}'.format(byte))([s_beta,outputs['beta']])

        sj = MultiLayer(name = 'multi_s_{}'.format(byte))([xor_sj_beta,outputs['alpha']])
        outputs['sj_{}'.format(byte)] = Softmax(name = 'output_sj_{}'.format(byte))(sj)     
        sj = InvSboxLayer(name = 'inv_s_{}'.format(byte))(sj)

        

        xor_tj_rin =  XorLayer(name = 'xor_tj_rin_{}'.format(byte))([t_rin,outputs['rin']])

        tj = MultiLayer(name = 'multi_t_{}'.format(byte))([xor_tj_rin,outputs['alpha']])
        outputs['tj_{}'.format(byte)] = Softmax(name = 'output_tj_{}'.format(byte))(tj)   
        
 
        kj = Add(name = 'add_{}'.format(byte))([tj,sj])
        outputs['kj_{}'.format(byte)] = Softmax(name = 'output_kj_{}'.format(byte))(kj)       
        metrics['kj_{}'.format(byte)] ='accuracy'
  
              
        
              


          
    losses = {}   

    weights = {}
    for k , v in outputs.items():
        
        losses[k] = 'categorical_crossentropy'
      
        weights[k] = 1 if not 'output' in k else 1
        


    model = Model(inputs = inputs_dict,outputs = outputs)    

    if summary:
        model.summary()
    return model  , losses  ,metrics  , weights















######################## ARCHITECTURE BUILDING ################################

def cnn_core(inputs_core,convolution_blocks , kernel_size,filters, strides , pooling_size):
    x = inputs_core
    for block in range(convolution_blocks):
        x = Conv1D(kernel_size=(kernel_size[block],), strides=strides, filters=filters, activation='selu', padding='same')(x)    
        x = BatchNormalization()(x)
        x = AveragePooling1D(pool_size = pooling_size)(x)
    
    output_layer = Flatten()(x) 

    return output_layer


def dense_core(inputs_core,dense_blocks,dense_units,bn = False,activated = False,name = ''):
    x = inputs_core
    
    for block in range(dense_blocks):
        x = Dense(dense_units, activation='selu')(x)   
        if bn:
            x = BatchNormalization()(x)
        
    if activated:
        output_layer = Dense(256,activation ='softmax' ,name = 'output_{}'.format(name))(x)  
    else:
        output_layer = Dense(256)(x)   
    return output_layer    



def dense_core_shared(inputs_core, shared_block = 1,non_shared_block = 1, units = 8, branches = 16,output_units = 32,precision = 'float32',split = False):
    flat = Flatten()(inputs_core)
    non_shared_branch = []
    if non_shared_block > 0:
        for branch in range(branches):
            x = inputs_core
            # for block in range(non_shared_block):
            if not split:
                x = flat
            x = Dense(units,activation ='selu',kernel_initializer=tf.keras.initializers.RandomUniform(seed=7))(x[:,:,branch] if split else x)
            #x = BatchNormalization()(x)
            non_shared_branch.append(tf.expand_dims(x,2))
        x = Concatenate(axis = 2)(non_shared_branch)
    else:
        x = inputs_core

    for block in range(shared_block):
        x = SharedWeightsDenseLayer(input_dim = x.shape[1],units = units,shares = branches)(x)        
        #x = BatchNormalization(axis = 1)(x)
    output_layer = SharedWeightsDenseLayer(input_dim = x.shape[1],units = output_units,activation = False,shares = branches,precision = precision)(x)   
    return output_layer 

#### Training high level function

def train_model(model_type,target_byte):
    epochs =25
    batch_size = 500
    n_traces = 450000
    
    model_t = 'model_{}'.format(model_type)   
    if model_type == 'hierarchical': 
        model_t = model_t + '_' + target_byte
        model , losses , metrics , weights  = model_hierarchical()
    if model_type == 'flat': 
        model_t = model_t + '_' + target_byte
        model , losses , metrics , weights  = model_flat()
    elif model_type == 'alpha':       
        model , losses , metrics , weights  = model_alpha_single()   
    elif model_type == 'rin':       
        model , losses , metrics , weights  = model_rin_single()             
    elif model_type == 'beta':       
        model , losses , metrics , weights  = model_beta_single()    
    elif model_type == 't1^rin' or  model_type == 's1^beta':   
        model_t = model_t + '_' + target_byte
        model , losses , metrics , weights  = model_intermediate_single()  
    else:
        print('You fucked up')
    




    learning_rates = [0.001,0.0001,0.00001]
    for cycle in range(3):
        callbacks = tf.keras.callbacks.ModelCheckpoint(
                                    filepath= MODEL_FOLDER+ model_t+'{}.h5'.format(cycle),
                                    save_weights_only=True,
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=True)
        optimizer = Adam(learning_rate=learning_rates[cycle])
        model.compile(loss=losses, optimizer=optimizer, metrics=metrics , loss_weights = weights)
        if model_type == 'hierarchical_now' or model_type == 'flat':
            X_profiling , validation_data = load_dataset_multi(flat = model_type =='flat',n_traces = n_traces,dataset = 'training',model_type = model_type) 
        else:
            X_profiling , validation_data = load_dataset(target_byte,n_traces = n_traces,dataset = 'training',model_type = model_type) 
        X_profiling = X_profiling.shuffle(len(X_profiling)).batch(batch_size) 
        validation_data = validation_data.batch(batch_size)
        history = model.fit(X_profiling, batch_size=batch_size, verbose = 1, epochs=epochs, validation_data=validation_data,callbacks =callbacks)
    print('Saved model ! ')
 
    file = open(METRICS_FOLDER+'history_training_'+(model_t ),'wb')
    pickle.dump(history.history,file)
    file.close()

    
    
    return model






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--HIERARCHICAL',   action="store_true", dest="HIERARCHICAL", help='Adding the masks to the labels', default=False)
    parser.add_argument('--FLAT',   action="store_true", dest="FLAT", help='Adding the masks to the labels', default=False)
        
    parser.add_argument('--SINGLE',   action="store_true", dest="SINGLE", help='Adding the masks to the labels', default=False)

    args            = parser.parse_args()
  
    HIERARCHICAL        = args.HIERARCHICAL
    FLAT = args.FLAT
    SINGLE        = args.SINGLE


    TARGETS = {}
    model_types = []
    if HIERARCHICAL:
        model_types  = ['hierarchical']
    elif FLAT:
        model_types  = ['flat']
    elif SINGLE:
        model_types = ['alpha','rin','beta','t1^rin','s1^beta']

    else:
        print('No training mode selected')



    for model_type in model_types:
        if model_type == 'hierarchical' or model_type == 'flat':
            
            process_eval = Process(target=train_model, args=(model_type,'all'))
            process_eval.start()
            process_eval.join()  
        else:
            for target_byte in VARIABLE_LIST[model_type]:
             
                process_eval = Process(target=train_model, args=(model_type,target_byte))
                process_eval.start()
                process_eval.join()  
                                


    print("$ Done !")
            
        
        
