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
from utility import MultiLayer , XorLayer , SharedWeightsDenseLayer , Add_Shares , InvSboxLayer

from utility import load_dataset, load_dataset_multi 
from tqdm import tqdm




seed = 7


tf.random.set_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

###########################################################################



class CombinedMetric(tf.keras.callbacks.Callback):
    def __init__(self):
        super(CombinedMetric, self).__init__()

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        somme = []
        for i in range(16):
            somme.append(logs["output_t_{}_accuracy".format(i)])
        logs['min_accuracy'] = min(somme)
        somme = []
        for i in range(16):
            somme.append(logs["val_output_t_{}_accuracy".format(i)])
        logs['val_min_accuracy'] = min(somme)

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
      
       
           
            losses_t = []

            for byte in range(0,16):
                losses_t.append(tf.keras.losses.categorical_crossentropy(y['output_t_{}'.format(byte)],y_pred['output_t_{}'.format(byte)])) 
            mean_t = tf.math.reduce_mean(losses_t)

            
            for byte in range(0,16):
                loss = loss + losses_t[byte] + tf.math.pow(losses_t[byte] - mean_t,2) 

            
            
           
            
                
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
         
       
        losses_t = []

        for byte in range(0,16):
            losses_t.append(tf.keras.losses.categorical_crossentropy(y['output_t_{}'.format(byte)],y_pred['output_t_{}'.format(byte)])) 
        mean_t = tf.math.reduce_mean(losses_t)

        
        for byte in range(0,16):
            loss = loss + losses_t[byte] + tf.math.pow(losses_t[byte] - mean_t,2) / 16

       
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        dict_metrics = {}
        for m in self.metrics:
            dict_metrics[m.name] = m.result()
        dict_metrics['loss'] = self.loss_tracker.result()
        return dict_metrics


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
    # s_beta_mj_core = dense_core_shared(block_core,output_units = 256)

    # t_rin_mj_core = dense_core_shared(block_core,output_units = 256)   
    
    metrics = {}
    for byte in range(16): 
        
        outputs['output_s_beta_{}'.format(byte)] = Softmax(name = 'output_s_beta_{}'.format(byte))(s_beta_core[:,:,byte])     
        metrics['output_s_beta_{}'.format(byte)] ='accuracy'

          
    losses = {}   

    weights = {}
    for k , v in outputs.items():
        if not 'sig' in k:
            losses[k] = 'categorical_crossentropy'
        else:
            losses[k] = 'binary_crossentropy'
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
        if not 'sig' in k:
            losses[k] = 'categorical_crossentropy'
        else:
            losses[k] = 'binary_crossentropy'
        weights[k] = 1 if not 'output' in k else 1
        


    model = Model(inputs = inputs_dict,outputs = outputs)    

    if summary:
        model.summary()
    return model  , losses  ,metrics  , weights

def model_permutations( learning_rate=0.001, classes=256 , name ='',summary = True):
    
    inputs_dict = {}
    outputs = {}
    

    inputs_permutations  = Input(shape = (93,16) ,name = 'inputs_permutations')
    inputs_dict['inputs_permutations'] = inputs_permutations  
    permutations_core = SharedWeightsDenseLayer(input_dim = inputs_permutations.shape[1],units = 64,shares = 16)(inputs_permutations)      
    permutations_core = BatchNormalization(axis = 1)(permutations_core)


    
    permutations_core = dense_core_shared(permutations_core,output_units = 16) 
    metrics = {}
    for byte in range(16):

        outputs['j_{}'.format(byte)] =  Softmax(name = 'output_j_{}'.format(byte))(permutations_core[:,:,byte])
        metrics['j_{}'.format(byte)] = 'accuracy'

          
    losses = {}   

    weights = {}
    for k , v in outputs.items():
        if not 'sig' in k:
            losses[k] = 'categorical_crossentropy'
        else:
            losses[k] = 'binary_crossentropy'
        weights[k] = 1 if not 'output' in k else 1
        


    model = Model(inputs = inputs_dict,outputs = outputs)    

    if summary:
        model.summary()
    return model  , losses  ,metrics  , weights

def model_permutations_single(learning_rate=0.001, classes=256 , name ='',summary = True):
    inputs_dict = {}
    outputs = {}
    

    inputs_permutations  = Input(shape = (93,16) ,name = 'inputs_permutations')
    inputs_dict['inputs_permutations'] = inputs_permutations  
    permutations_core = SharedWeightsDenseLayer(input_dim = inputs_permutations.shape[1],units = 64,shares = 16)(inputs_permutations)      
    permutations_core = BatchNormalization(axis = 1)(permutations_core)
    permutations_core = Flatten()(permutations_core)

    permutations_core = Dense( 8,activation = 'selu')(permutations_core)      
    permutations_core = BatchNormalization()(permutations_core)

    permutations_core = Dense( 8,activation = 'selu')(permutations_core)      
    permutations_core = BatchNormalization()(permutations_core)
    
    permutations_core = Dense( 16,activation = 'softmax')(permutations_core)      

    metrics = {}

    outputs['output'] =  permutations_core
    metrics['output'] = 'accuracy'

          
    losses = {}   

    weights = {}
    for k , v in outputs.items():
        if not 'sig' in k:
            losses[k] = 'categorical_crossentropy'
        else:
            losses[k] = 'binary_crossentropy'
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
        if not 'sig' in k:
            losses[k] = 'categorical_crossentropy'
        else:
            losses[k] = 'binary_crossentropy'
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
        if not 'sig' in k:
            losses[k] = 'categorical_crossentropy'
        else:
            losses[k] = 'binary_crossentropy'
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
        if not 'sig' in k:
            losses[k] = 'categorical_crossentropy'
        else:
            losses[k] = 'binary_crossentropy'
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
        if not 'sig' in k:
            losses[k] = 'categorical_crossentropy'
        else:
            losses[k] = 'binary_crossentropy'
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
        # outputs['output_s_beta_mj_{}'.format(byte)] =  Softmax(name = 'output_s_beta_mj_{}'.format(byte),dtype = tf.float64)(s_beta_mj_core[:,:,byte])
        outputs['t_rin_{}'.format(byte)] =  Softmax(name = 'output_t_rin_{}'.format(byte))(t_rin)

    




        
              


          
    losses = {}   
    # for k , v in outputs.items():
    #     if  'sig' in k:
    #         losses[k] = 'binary_crossentropy'
    #     elif 'alpha' in k or 'beta' in k or 'rin' in k:
    #         losses[k] = 'categorical_crossentropy'



    # model = Multi_Model(inputs = inputs_dict,outputs = outputs)
    weights = {}
    metrics = {}
    for k , v in outputs.items():
        if not 'sig' in k:
            losses[k] = 'categorical_crossentropy'
        else:
            losses[k] = 'binary_crossentropy'
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
    # block_core = SharedWeightsDenseLayer(input_dim = inputs_block.shape[1],units = 64,shares = 16)(inputs_block)      
    # block_core = BatchNormalization(axis = 1)(block_core)
    # block_core = Flatten()(block_core)
    
    # plaintext = Input(shape=(16,256))
    # inputs_dict['plaintexts'] = plaintext
    
    # inputs_permutations  = Input(shape = (93,16) ,name = 'inputs_permutations')
    # inputs_dict['inputs_permutations'] = inputs_permutations  
    # permutations_core = SharedWeightsDenseLayer(input_dim = inputs_permutations.shape[1],units = 64,shares = 16)(inputs_permutations)      
    # permutations_core = BatchNormalization(axis = 1)(permutations_core)
    
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
        # outputs['output_s_beta_mj_{}'.format(byte)] =  Softmax(name = 'output_s_beta_mj_{}'.format(byte),dtype = tf.float64)(s_beta_mj_core[:,:,byte])
        outputs['t_rin_{}'.format(byte)] =  Softmax(name = 'output_t_rin_{}'.format(byte))(t_rin)
        # outputs['output_t_rin_mj_{}'.format(byte)] =  Softmax(name = 'output_t_rin_mj_{}'.format(byte),dtype = tf.float64)(t_rin_mj_core[:,:,byte])
        # outputs['output_mj_{}'.format(byte)] =  Softmax(name = 'output_mj_{}'.format(byte))(mj_core[:,:,byte])


 

        # mj = Add()([mj_from_m_all[:,:,byte]  , mj_core[:,:,byte] ])
        # # mj = Softmax(name = 'output_mj_{}'.format(byte))(mj)
        # mj = Softmax()(mj)
        
        # beta_mj = XorLayer(name = 'xor_beta_mj_{}'.format(byte))([mj,outputs['output_beta']])
        # rin_mj = XorLayer(name = 'xor_rin_mj_{}'.format(byte))([mj,outputs['output_rin']])

        # xor_sj_mj =  XorLayer(name = 'xor_sj_mj_{}'.format(byte))([s_mj_core[:,:,byte],mj])
        # xor_sj_beta_mj =  XorLayer(name = 'xor_sj_beta_mj_{}'.format(byte))([s_beta_mj_core[:,:,byte],beta_mj])
        xor_sj_beta =  XorLayer(name = 'xor_sj_beta_{}'.format(byte))([s_beta,outputs['beta']])
        # alpha_sj = Add()([xor_sj_mj,xor_sj_beta_mj,xor_sj_beta])
        
        sj = MultiLayer(name = 'multi_s_{}'.format(byte))([xor_sj_beta,outputs['alpha']])
        outputs['sj_{}'.format(byte)] = Softmax(name = 'output_sj_{}'.format(byte))(sj)     
        sj = InvSboxLayer(name = 'inv_s_{}'.format(byte))(sj)
        #sj = tf.keras.layers.Activation('sigmoid')(sj)
        
        
        
        # xor_tj_mj =  XorLayer(name = 'xor_tj_mj_{}'.format(byte))([t_mj_core[:,:,byte],mj])
        # xor_tj_rin_mj =  XorLayer(name = 'xor_tj_rin_mj_{}'.format(byte))([t_rin_mj_core[:,:,byte],rin_mj])
        xor_tj_rin =  XorLayer(name = 'xor_tj_rin_{}'.format(byte))([t_rin,outputs['rin']])
        # alpha_tj = Add()([xor_tj_mj,xor_tj_rin_mj,xor_tj_rin])
        tj = MultiLayer(name = 'multi_t_{}'.format(byte))([xor_tj_rin,outputs['alpha']])
        outputs['tj_{}'.format(byte)] = Softmax(name = 'output_tj_{}'.format(byte))(tj)   
        
        # tj = tf.keras.layers.Activation('sigmoid')(tj)
        kj = Add(name = 'add_{}'.format(byte))([tj,sj])
        outputs['kj_{}'.format(byte)] = Softmax(name = 'output_kj_{}'.format(byte))(kj)       
        metrics['kj_{}'.format(byte)] ='accuracy'
  
              
        
              


          
    losses = {}   
    # for k , v in outputs.items():
    #     if  'sig' in k:
    #         losses[k] = 'binary_crossentropy'
    #     elif 'alpha' in k or 'beta' in k or 'rin' in k:
    #         losses[k] = 'categorical_crossentropy'



    # model = Multi_Model(inputs = inputs_dict,outputs = outputs)
    weights = {}
    for k , v in outputs.items():
        if not 'sig' in k:
            losses[k] = 'categorical_crossentropy'
        else:
            losses[k] = 'binary_crossentropy'
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
    if model_type == 'hierarchical_now': 
        model_t = model_t + '_' + target_byte
        model , losses , metrics , weights  = model_hierarchical()
    if model_type == 'flat': 
        model_t = model_t + '_' + target_byte
        model , losses , metrics , weights  = model_flat()
    # elif model_type == 'sbox_input':       
    #     model , losses , metrics , weights  = model_sbox_input()       
    # elif model_type == 'sbox_output':       
    #     model , losses , metrics , weights  = model_sbox_output()  
    # elif model_type == 'permutations':       
    #     model , losses , metrics , weights  = model_permutations()  
    elif model_type == 'alpha':       
        model , losses , metrics , weights  = model_alpha_single()   
    elif model_type == 'rin':       
        model , losses , metrics , weights  = model_rin_single()             
    elif model_type == 'beta':       
        model , losses , metrics , weights  = model_beta_single()    
    elif model_type == 't1^rin' or  model_type == 's1^beta':   
        model_t = model_t + '_' + target_byte
        model , losses , metrics , weights  = model_intermediate_single()  
    elif model_type == 'p':   
        model_t = model_t + '_' + target_byte
        model , losses , metrics , weights  = model_permutations_single()  
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
        model_types  = ['hierarchical_now']
    elif FLAT:
        model_types  = ['flat']
    elif SINGLE:
        model_types = ['t1^rin']

    else:
        print('No training mode selected')

    # for model_random in tqdm(range(25)):
    #     convolution_blocks = np.random.randint(1,3)
    #     kernel_size = sorted(np.random.randint(4,32,size = convolution_blocks))       
    #     filters = np.random.randint(3,16)
    #     pooling_size = np.random.randint(2,5)
    #     dense_blocks = np.random.randint(1,5)
    #     dense_units = np.random.randint(64,512)

    for model_type in model_types:
        if model_type == 'hierarchical_now' or model_type == 'flat':
            
            process_eval = Process(target=train_model, args=(model_type,'all'))
            process_eval.start()
            process_eval.join()  
        else:
            for target_byte in VARIABLE_LIST[model_type]:
                print(target_byte)
                process_eval = Process(target=train_model, args=(model_type,target_byte))
                process_eval.start()
                process_eval.join()  
                if 't002' in target_byte:
                    break                                  


    print("$ Done !")
            
        
        
