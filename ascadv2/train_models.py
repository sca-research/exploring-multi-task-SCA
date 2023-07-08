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
from utility import  METRICS_FOLDER , MODEL_FOLDER

# import custom layers
from utility import MultiLayer , XorLayer , SharedWeightsDenseLayer , Add_Shares , InvSboxLayer

from utility import load_dataset, load_dataset_multi , load_dataset_hierarchical
from tqdm import tqdm

import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()


seed = 7


tf.random.set_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

###########################################################################

class Cross_Model(tf.keras.Model):

    
    def __init__(self,inputs,outputs):
        
        super(Cross_Model, self).__init__(inputs = inputs,outputs = outputs)
        

        #self.snr_layer = snr_layer
        all_maps = np.load('utils/xor_mapping.npy')
        mapping1 = []
        mapping2 = []
        for classe in range(256):
            mapped = np.where(all_maps[classe] == 1)
            mapping1.append(mapped[0])
            mapping2.append(mapped[1])
        self.xor_mapping1 = np.array(mapping1)
        self.xor_mapping2 = np.array(mapping2) 
        self.loss_tracker = tf.keras.metrics.Mean(name= 'loss')


                

    def train_step(self, data):
        x, y = data
        labels_mask = None
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(
                y,
                y_pred,
                regularization_losses=self.losses,
            )
            res = {}
            for byte in range(16):
                target = tf.cast(y['output_{}'.format(byte)],tf.float32)
                predictions = y_pred['output_masked_{}'.format(byte)]
                pred_true = tnp.asarray(target)[:,self.xor_mapping1]
                pred_predictions = tnp.asarray(predictions)[:,self.xor_mapping2]
                res[byte] = tf.reduce_sum(tf.multiply(pred_true,pred_predictions),axis =2)   
                if labels_mask is None:
                    labels_mask = res[byte]
                else:
                    labels_mask = labels_mask + res[byte]

            mask_loss = tf.keras.losses.categorical_crossentropy(labels_mask/16,y_pred['output_mask']) 
            loss = loss +  mask_loss 
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        

        # Compute our own metrics
        self.loss_tracker.update_state(loss)

        #self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
      
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
        labels_mask = None
        y_pred = self(x, training=False)
        loss = self.compiled_loss(
            y,
            y_pred,
            regularization_losses=self.losses,
        )
        res = {}
        for byte in range(16):
            target = tf.cast(y['output_{}'.format(byte)],tf.float32)
            predictions = y_pred['output_masked_{}'.format(byte)]
            pred_true = tnp.asarray(target)[:,self.xor_mapping1]
            pred_predictions = tnp.asarray(predictions)[:,self.xor_mapping2]
            res[byte] = tf.reduce_sum(tf.multiply(pred_true,pred_predictions),axis =2)   
            if labels_mask is None:
                labels_mask = res[byte]
            else:
                labels_mask = labels_mask + res[byte]

        mask_loss = tf.keras.losses.categorical_crossentropy(labels_mask/16,y_pred['output_mask'])  
        loss = loss + mask_loss  

        
        # Update val metrics
        self.loss_tracker.update_state(loss)
    
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        dict_metrics = {}
        for m in self.metrics:
            dict_metrics[m.name] = m.result()
        dict_metrics['loss'] = self.loss_tracker.result()
        return dict_metrics


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
    
    def __init__(self,inputs,outputs,s = False):
        
        super(Multi_Model, self).__init__(inputs = inputs,outputs = outputs)
        
        self.loss_tracker = tf.keras.metrics.Mean(name= 'loss')  
        self.s = s


    
    def train_step(self, data):
        x, y = data
   
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(
                y,
                y_pred,
                regularization_losses=self.losses,
            )
      
       
           
            losses_m = []
            losses_j = []
            losses_sj = []
            losses_tj = []
            for byte in range(0,16):
 
                losses_tj.append(tf.keras.losses.categorical_crossentropy(y['output_{}'.format(byte)],y_pred['output_{}'.format(byte)])) 

            mean_tj = tf.math.reduce_mean(losses_tj)
            
            for byte in range(0,16):

                loss = loss  + tf.math.pow(losses_tj[byte] - mean_tj,2) / 16
            
            
           
            
                
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
         
       
        losses_m = []
        losses_j = []
        losses_sj = []
        losses_tj = []
        for byte in range(0,16):

            losses_tj.append(tf.keras.losses.categorical_crossentropy(y['output_{}'.format(byte)],y_pred['output_{}'.format(byte)])) 

        mean_tj = tf.math.reduce_mean(losses_tj)
        
        for byte in range(0,16):

            loss = loss  + tf.math.pow(losses_tj[byte] - mean_tj,2) / 16
       
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        dict_metrics = {}
        for m in self.metrics:
            dict_metrics[m.name] = m.result()
        dict_metrics['loss'] = self.loss_tracker.result()
        return dict_metrics

class Multi_Model_2(tf.keras.Model):
    
    def __init__(self,inputs,outputs,s = False):
        
        super(Multi_Model_2, self).__init__(inputs = inputs,outputs = outputs)
        
        self.loss_tracker = tf.keras.metrics.Mean(name= 'loss')  
        self.s = s


    
    def train_step(self, data):
        x, y = data
   
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(
                y,
                y_pred,
                regularization_losses=self.losses,
            )
      
       
           
            losses_m = []
            losses_j = []
            losses_sj = []
            losses_tj = []
            for byte in range(0,16):
 
                losses_tj.append(tf.keras.losses.categorical_crossentropy(y['output_tj_{}'.format(byte)],y_pred['output_tj_{}'.format(byte)])) 
                losses_sj.append(tf.keras.losses.categorical_crossentropy(y['output_sj_{}'.format(byte)],y_pred['output_sj_{}'.format(byte)])) 

            mean_tj = tf.math.reduce_mean(losses_tj)
            mean_sj = tf.math.reduce_mean(losses_sj)
            
            for byte in range(0,16):

                loss = loss  + tf.math.pow(losses_tj[byte] - mean_tj,2) / 16
                loss = loss  + tf.math.pow(losses_sj[byte] - mean_sj,2) / 16
            
            
           
            
                
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
         
       
        losses_m = []
        losses_j = []
        losses_sj = []
        losses_tj = []
        for byte in range(0,16):
 
            losses_tj.append(tf.keras.losses.categorical_crossentropy(y['output_tj_{}'.format(byte)],y_pred['output_tj_{}'.format(byte)])) 
            losses_sj.append(tf.keras.losses.categorical_crossentropy(y['output_sj_{}'.format(byte)],y_pred['output_sj_{}'.format(byte)])) 

        mean_tj = tf.math.reduce_mean(losses_tj)
        mean_sj = tf.math.reduce_mean(losses_sj)
        
        for byte in range(0,16):

            loss = loss  + tf.math.pow(losses_tj[byte] - mean_tj,2) / 16
            loss = loss  + tf.math.pow(losses_sj[byte] - mean_sj,2) / 16
       
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



def model_hierarchical(learning_rate=0.001, classes=256,shared = False , name ='',summary = True,permutations = False):
    
    inputs_dict = {}
    outputs = {}
    
    inputs_block  = Input(shape = (93,16) ,name = 'inputs_block')
    inputs_dict['inputs_block'] = inputs_block 

    plaintexts = Input(shape = (16,256) ,name = 'plaintexts')
    inputs_dict['plaintexts'] = plaintexts
    
    inputs_alpha  = Input(shape = (2000,1) ,name = 'inputs_alpha')
    inputs_dict['inputs_alpha'] = inputs_alpha 
    alpha_core = cnn_core(inputs_alpha,1,[32],16,10,5)
    alpha_core = dense_core(alpha_core,1,64,activated = True,name = 'alpha')
    # outputs['output_alpha'] = alpha_core
    
    inputs_rin  = Input(shape = (1000,1) ,name = 'inputs_rin')
    inputs_dict['inputs_rin'] = inputs_rin  
    rin_core = cnn_core(inputs_rin,1,[32],16,5,5)
    rin_core = dense_core(rin_core,1,64,activated = True,name = 'rin')
    # outputs['output_rin'] = rin_core
    
    inputs_beta  = Input(shape = (200,1) ,name = 'inputs_beta')    
    inputs_dict['inputs_beta'] = inputs_beta  
    beta_core = cnn_core(inputs_beta,1,[32],16,1,5)
    beta_core = dense_core(beta_core,1,64,activated = True,name = 'beta')
    # outputs['output_beta'] = beta_core


    metrics = {}
    s_branch = []
    t_branch = []
    branches = []
    for byte in range(16):
        block_core = SharedWeightsDenseLayer(input_dim = inputs_block.shape[1],units = 64,shares = 16)(inputs_block)      
        block_core = BatchNormalization(axis = 1)(block_core)
        block_core = Flatten()(block_core)
        branches.append(tf.expand_dims(block_core,2))
    block_core = Concatenate(axis = 2)(branches)

    if shared:
        s_branch = dense_core_shared(block_core,shared_block = 1,non_shared_block = 1,units = 8,output_units=256,split = True)
        t_branch = dense_core_shared(block_core,shared_block = 1,non_shared_block = 1,units = 8,output_units=256,split = True)

    else:
        for byte in range(16):   
            s_beta_core = dense_core(block_core[:,:,byte],2,8)
            s_branch.append(tf.expand_dims(s_beta_core,2))
            # s_beta_mj_core = dense_core_shared(block_core,output_units = 256)
            t_rin_core = dense_core(block_core[:,:,byte],2,8)
            t_branch.append(tf.expand_dims(t_rin_core,2))
        s_branch = Concatenate(axis = 2)(s_branch)
        t_branch = Concatenate(axis = 2)(t_branch)
        # mj = Add()([mj_from_m_all[:,:,byte]  , mj_core[:,:,byte] ])
        # # mj = Softmax(name = 'output_mj_{}'.format(byte))(mj)
        # mj = Softmax()(mj)
    for byte in range(16):     

        xor_sj_beta =  XorLayer(name = 'xor_sj_beta_{}'.format(byte))([s_branch[:,:,byte],beta_core])
        sj = MultiLayer(name = 'multi_s_{}'.format(byte))([xor_sj_beta,alpha_core])
        outputs['output_sj_{}'.format(byte)] = Softmax(name = 'output_sj_{}'.format(byte))(sj)     

        xor_tj_rin =  XorLayer(name = 'xor_tj_rin_{}'.format(byte))([t_branch[:,:,byte],rin_core])
        tj = MultiLayer(name = 'multi_t_{}'.format(byte))([xor_tj_rin,alpha_core])
        outputs['output_tj_{}'.format(byte)] = Softmax(name = 'output_tj_{}'.format(byte))(tj)     
        
        tj_from_sj = InvSboxLayer(name = 'inv_{}'.format(byte))(sj)
        
        add = Add()([tj_from_sj,tj])
        xor_plaintext = XorLayer(name = 'xor_p_{}'.format(byte))([add,plaintexts[:,byte]])
        outputs['output_kj_{}'.format(byte)] = Softmax(name = 'output_kj_{}'.format(byte))(xor_plaintext)    
        
        
        metrics['output_kj_{}'.format(byte)] ='accuracy'
        
        


          
    losses = {}   
    # for k , v in outputs.items():
    #     if  'sig' in k:
    #         losses[k] = 'binary_crossentropy'
    #     elif 'alpha' in k or 'beta' in k or 'rin' in k:
    #         losses[k] = 'categorical_crossentropy'



    # model = Multi_Model(inputs = inputs_dict,outputs = outputs)
    for k , v in outputs.items():
        if not 'sig' in k:
            losses[k] = 'categorical_crossentropy'
        else:
            losses[k] = 'binary_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')    

    if summary:
        model.summary()
    return model  , losses  ,metrics 


def model_multi(learning_rate=0.001, classes=256,shared = False , name ='',summary = True,permutations = False,multi_model = False):
    
    inputs_dict = {}
    outputs = {}
    
    inputs_block  = Input(shape = (93,16) ,name = 'inputs_block')
    inputs_dict['inputs_block'] = inputs_block 

    if permutations:
        inputs_permutations  = Input(shape = (93,16) ,name = 'inputs_permutations')
        inputs_dict['inputs_permutations'] = inputs_permutations  
        permutations_core = cnn_core(inputs_permutations,1,[13],16,1,2) 
    
    inputs_alpha  = Input(shape = (2000,1) ,name = 'inputs_alpha')
    inputs_dict['inputs_alpha'] = inputs_alpha 
    alpha_core = cnn_core(inputs_alpha,1,[32],16,10,5)
    alpha_core = dense_core(alpha_core,1,64,activated = True,name = 'alpha')
    # outputs['output_alpha'] = alpha_core
    
    inputs_rin  = Input(shape = (1000,1) ,name = 'inputs_rin')
    inputs_dict['inputs_rin'] = inputs_rin  
    rin_core = cnn_core(inputs_rin,1,[32],16,5,5)
    rin_core = dense_core(rin_core,1,64,activated = True,name = 'rin')
    # outputs['output_rin'] = rin_core
    
    inputs_beta  = Input(shape = (200,1) ,name = 'inputs_beta')    
    inputs_dict['inputs_beta'] = inputs_beta  
    beta_core = cnn_core(inputs_beta,1,[32],16,1,5)
    beta_core = dense_core(beta_core,1,64,activated = True,name = 'beta')
    # outputs['output_beta'] = beta_core

    if permutations:
        tj_matrix = []
        permutations_core = dense_core_shared(permutations_core,output_units = 16,split = True) 
        permutations_matrix = []
        for byte in range(16):
            permutations_matrix.append(tf.expand_dims(Softmax(name = 'output_j_{}'.format(byte))(permutations_core[:,:,byte]),2))
        permutations_matrix = Concatenate(axis = 2)(permutations_matrix)
    # # mj_from_m_all = tf.matmul(m_core,tf.transpose(permutations_matrix,[0,2,1]))
    metrics = {}
    s_branch = []
    t_branch = []
    branches = []
    for byte in range(16):
        block_core = SharedWeightsDenseLayer(input_dim = inputs_block.shape[1],units = 64,shares = 16)(inputs_block)      
        block_core = BatchNormalization(axis = 1)(block_core)
        block_core = Flatten()(block_core)
        branches.append(tf.expand_dims(block_core,2))
    block_core = Concatenate(axis = 2)(branches)

    if shared:
        s_branch = dense_core_shared(block_core,shared_block = 1,non_shared_block = 1,units = 8,output_units=256,split = True)
        t_branch = dense_core_shared(block_core,shared_block = 1,non_shared_block = 1,units = 8,output_units=256,split = True)

    else:
        for byte in range(16):   
            s_beta_core = dense_core(block_core[:,:,byte],2,8)
            s_branch.append(tf.expand_dims(s_beta_core,2))
            # s_beta_mj_core = dense_core_shared(block_core,output_units = 256)
            t_rin_core = dense_core(block_core[:,:,byte],2,8)
            t_branch.append(tf.expand_dims(t_rin_core,2))
        s_branch = Concatenate(axis = 2)(s_branch)
        t_branch = Concatenate(axis = 2)(t_branch)
        # mj = Add()([mj_from_m_all[:,:,byte]  , mj_core[:,:,byte] ])
        # # mj = Softmax(name = 'output_mj_{}'.format(byte))(mj)
        # mj = Softmax()(mj)
    for byte in range(16):     

        xor_sj_beta =  XorLayer(name = 'xor_sj_beta_{}'.format(byte))([s_branch[:,:,byte],beta_core])
        sj = MultiLayer(name = 'multi_s_{}'.format(byte))([xor_sj_beta,alpha_core])
        outputs['output_sj_{}'.format(byte)] = Softmax(name = 'output_sj_{}'.format(byte))(sj)     

        xor_tj_rin =  XorLayer(name = 'xor_tj_rin_{}'.format(byte))([t_branch[:,:,byte],rin_core])
        tj = MultiLayer(name = 'multi_t_{}'.format(byte))([xor_tj_rin,alpha_core])

        outputs['output_tj_{}'.format(byte)] = Softmax(name = 'output_tj_{}'.format(byte))(tj)     
        metrics['output_tj_{}'.format(byte)] ='accuracy'


          
    losses = {}   
    # for k , v in outputs.items():
    #     if  'sig' in k:
    #         losses[k] = 'binary_crossentropy'
    #     elif 'alpha' in k or 'beta' in k or 'rin' in k:
    #         losses[k] = 'categorical_crossentropy'



    # 
    for k , v in outputs.items():
        if not 'sig' in k:
            losses[k] = 'categorical_crossentropy'
        else:
            losses[k] = 'binary_crossentropy'

    if multi_model:
        model = Multi_Model_2(inputs = inputs_dict,outputs = outputs)
    else:
        model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')    

    if summary:
        model.summary()
    return model  , losses  ,metrics 




def model_single_task(s = False, t = False,alpha_known = True, summary = True):
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
        alpha_core = cnn_core(inputs_alpha,1,[32],16,10,5)
        alpha = dense_core(alpha_core,1,64,activated = True,name = 'alpha')
    # outputs['output_alpha'] = alpha_core
    

    if s:
        inputs_beta  = Input(shape = (200,1) ,name = 'inputs_beta')    
        inputs_dict['inputs_beta'] = inputs_beta  
        beta_core = cnn_core(inputs_beta,1,[32],16,1,5)
        mask_core = dense_core(beta_core,1,64,activated = True,name = 'beta')
    if t:
  
    
        inputs_rin  = Input(shape = (1000,1) ,name = 'inputs_rin')
        inputs_dict['inputs_rin'] = inputs_rin  
        rin_core = cnn_core(inputs_rin,1,[32],16,5,5)
        mask_core = dense_core(rin_core,1,64,activated = True,name = 'rin')
        # outputs['output_rin'] = rin_core
        

    metrics = {}

    block_core = SharedWeightsDenseLayer(input_dim = inputs_block.shape[1],units = 64,shares = 16)(inputs_block)      
    block_core = BatchNormalization(axis = 1)(block_core)
    block_core = Flatten()(block_core)
    intermediate_core = dense_core(block_core,2,8)
           
     
    xor =  XorLayer(name = 'xor')([intermediate_core,mask_core])
    mult = MultiLayer(name = 'multi')([xor,alpha])

    outputs['output'] = Softmax(name = 'output')(mult)     
    metrics['output'] ='accuracy'
    losses = {}
    losses['output'] = 'categorical_crossentropy'
    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_single')    
    return model  , losses  ,metrics 
def model_multi_task_s_only(permutations = False,multi_model = False,cross_model = False,shared = False,known_alpha = False,summary = True):
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
        alpha_core = cnn_core(inputs_alpha,1,[32],16,10,5)
        alpha = dense_core(alpha_core,1,64,activated = True,name = 'alpha')
        outputs['output_alpha'] = alpha
    

    
    inputs_beta  = Input(shape = (200,1) ,name = 'inputs_beta')    
    inputs_dict['inputs_beta'] = inputs_beta  
    beta_core = cnn_core(inputs_beta,1,[32],16,1,5)
    beta_core = dense_core(beta_core,1,64,activated = True,name = 'beta')
    if cross_model :
        outputs['output_mask'] = beta_core
    # outputs['output_beta'] = beta_core


    metrics = {}
    s_branch = []
    t_branch = []
    branches = []
    for byte in range(16):
        block_core = SharedWeightsDenseLayer(input_dim = inputs_block.shape[1],units = 64,shares = 16)(inputs_block)      
        block_core = BatchNormalization(axis = 1)(block_core)
        block_core = Flatten()(block_core)
        branches.append(tf.expand_dims(block_core,2))
    block_core = Concatenate(axis = 2)(branches)

    if shared:
        s_branch = dense_core_shared(block_core,shared_block = 1,non_shared_block = 1,units = 8,output_units=256,split = True)
    else:
        for byte in range(16):   
            s_beta_core = dense_core(block_core[:,:,byte],2,8)
            s_branch.append(tf.expand_dims(s_beta_core,2))

        s_branch = Concatenate(axis = 2)(s_branch)

    for byte in range(16):   
        if cross_model:
            soft_byte = Softmax()(s_branch[:,:,byte])
            outputs['output_masked_{}'.format(byte)] = soft_byte
        xor_sj_beta =  XorLayer(name = 'xor_sj_beta_{}'.format(byte))([s_branch[:,:,byte],beta_core])
        sj = MultiLayer(name = 'multi_s_{}'.format(byte))([xor_sj_beta,alpha])
        outputs['output_{}'.format(byte)] = Softmax(name = 'output_sj_{}'.format(byte))(sj)     
        metrics['output_{}'.format(byte)] ='accuracy'

          
    losses = {}   
    # for k , v in outputs.items():
    #     if  'sig' in k:
    #         losses[k] = 'binary_crossentropy'
    #     elif 'alpha' in k or 'beta' in k or 'rin' in k:
    #         losses[k] = 'categorical_crossentropy'



    # model = Multi_Model(inputs = inputs_dict,outputs = outputs)
    for k , v in outputs.items():
    
        losses[k] = 'categorical_crossentropy'
    

    if not cross_model:


        if not multi_model :
            model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')    
        else:
            model = Multi_Model(inputs = inputs_dict,outputs = outputs)
    else:
        model = Cross_Model(inputs = inputs_dict,outputs = outputs)
    if summary:
        model.summary()
    return model  , losses  ,metrics 


def model_multi_task_t_only(permutations = False,shared = False,multi_model = False,cross_model = False,known_alpha = False,summary = True):
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
        alpha_core = cnn_core(inputs_alpha,1,[32],16,10,5)
        alpha = dense_core(alpha_core,1,64,activated = True,name = 'alpha')
        outputs['output_alpha'] = alpha
    
    inputs_rin  = Input(shape = (1000,1) ,name = 'inputs_rin')
    inputs_dict['inputs_rin'] = inputs_rin  
    rin_core = cnn_core(inputs_rin,1,[32],16,5,5)
    rin_core = dense_core(rin_core,1,64,activated = True,name = 'rin')
    if cross_model:
        outputs['output_mask'] = rin_core
    # outputs['output_rin'] = rin_core
    

    metrics = {}
    s_branch = []
    t_branch = []
    branches = []
    for byte in range(16):
        block_core = SharedWeightsDenseLayer(input_dim = inputs_block.shape[1],units = 64,shares = 16)(inputs_block)      
        block_core = BatchNormalization(axis = 1)(block_core)
        block_core = Flatten()(block_core)
        branches.append(tf.expand_dims(block_core,2))
    block_core = Concatenate(axis = 2)(branches)

    if shared:
        t_branch = dense_core_shared(block_core,shared_block = 1,non_shared_block = 1,units = 8,output_units=256,split = True)

    else:
        for byte in range(16):   

            t_rin_core = dense_core(block_core[:,:,byte],2,8)
            t_branch.append(tf.expand_dims(t_rin_core,2))
     
        t_branch = Concatenate(axis = 2)(t_branch)
        # mj = Add()([mj_from_m_all[:,:,byte]  , mj_core[:,:,byte] ])
        # # mj = Softmax(name = 'output_mj_{}'.format(byte))(mj)
        # mj = Softmax()(mj)
    for byte in range(16):     
        if cross_model:
            soft_byte = Softmax()(t_branch[:,:,byte])
            outputs['output_masked_{}'.format(byte)] = soft_byte
        xor_tj_rin =  XorLayer(name = 'xor_tj_rin_{}'.format(byte))([t_branch[:,:,byte],rin_core])
        tj = MultiLayer(name = 'multi_t_{}'.format(byte))([xor_tj_rin,alpha])

        outputs['output_{}'.format(byte)] = Softmax(name = 'output_tj_{}'.format(byte))(tj)     
        metrics['output_{}'.format(byte)] ='accuracy'


          
    losses = {}   
    # for k , v in outputs.items():
    #     if  'sig' in k:
    #         losses[k] = 'binary_crossentropy'
    #     elif 'alpha' in k or 'beta' in k or 'rin' in k:
    #         losses[k] = 'categorical_crossentropy'



    # model = Multi_Model(inputs = inputs_dict,outputs = outputs)
    for k , v in outputs.items():
    
        losses[k] = 'categorical_crossentropy'
    

    if not cross_model:


        if not multi_model :
            model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')    
        else:
            model = Multi_Model(inputs = inputs_dict,outputs = outputs)
    else:
        model = Cross_Model(inputs = inputs_dict,outputs = outputs)
    if summary:
        model.summary()
    return model  , losses  ,metrics 






######################## ARCHITECTURE BUILDING ################################

def cnn_core(inputs_core,convolution_blocks , kernel_size,filters, strides , pooling_size):
    x = inputs_core
    for block in range(convolution_blocks):
        x = Conv1D(kernel_size=(kernel_size[block],), strides=strides, filters=filters, activation='selu', padding='same')(x)    
        x = BatchNormalization()(x)
        x = AveragePooling1D(pool_size = pooling_size)(x)
    
    output_layer = Flatten()(x) 

    return output_layer


def dense_core(inputs_core,dense_blocks,dense_units,activated = False,name = ''):
    x = inputs_core
    
    for block in range(dense_blocks):
        x = Dense(dense_units, activation='selu',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)        
        x = BatchNormalization()(x)
        
    if activated:
        output_layer = Dense(256,activation ='softmax' ,name = 'output_{}'.format(name),kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)  
    else:
        output_layer = Dense(256,kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)   
    return output_layer    

def dense_core_shared(inputs_core, shared_block = 1,non_shared_block = 1, units = 8, branches = 16,output_units = 32,precision = 'float32',split = False):
    non_shared_branch = []
    if non_shared_block > 0:
        for branch in range(branches):
            x = inputs_core
            # for block in range(non_shared_block):
            x = Dense(units,activation ='selu',kernel_initializer=tf.keras.initializers.RandomUniform(seed=7))(x[:,:,branch] if split else x)
            x = BatchNormalization()(x)
            non_shared_branch.append(tf.expand_dims(x,2))
        x = Concatenate(axis = 2)(non_shared_branch)
    else:
        x = inputs_core

    for block in range(shared_block):
        x = SharedWeightsDenseLayer(input_dim = x.shape[1],units = units,shares = branches)(x)        
        x = BatchNormalization(axis = 1)(x)
    output_layer = SharedWeightsDenseLayer(input_dim = x.shape[1],units = output_units,activation = False,shares = branches,precision = precision)(x)   
    return output_layer 

#### Training high level function

def train_model(multi_model,shared,training_type,byte):
    permutations = False
    batch_size = 500
    n_traces = 450000
    known_alpha = 'first' in training_type
    

    
    model_t = '{}_{}_{}{}'.format('multi_model' if multi_model else 'model',training_type, 'shared' if shared else 'nshared','_'+str(byte) if not (byte is None) else '')
    
    if training_type == 'multi':     
        model , losses , metrics  = model_multi(permutations = permutations,shared = shared,multi_model = multi_model)
    elif training_type == 'hierarchical':
        model , losses , metrics  = model_hierarchical(shared = shared)        
    elif training_type == 'multi_t':
        model , losses , metrics  = model_multi_task_t_only(permutations = permutations,shared = shared,multi_model = multi_model, known_alpha = known_alpha)
    elif training_type == 'multi_s':       
        model , losses , metrics  = model_multi_task_s_only(permutations = permutations,shared = shared,multi_model = multi_model, known_alpha = known_alpha)
    elif 'single' in training_type:
        model , losses , metrics  = model_single_task(s = 'single_s' in training_type, t = 'single_t' in training_type, alpha_known = 'first' in training_type)
    else:
        print('')



    learning_rates = [0.001,0.0001,0.00001]
    epochs  = [25,25,25]
    for cycle in range(3):
        callbacks = tf.keras.callbacks.ModelCheckpoint(
                                    filepath= MODEL_FOLDER+ model_t+'{}.h5'.format(cycle),
                                    save_weights_only=True,
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=True)
        optimizer = Adam(learning_rate=learning_rates[cycle])
        model.compile(loss=losses, optimizer=optimizer, metrics=metrics)
        if 'multi' in training_type:           
            X_profiling , validation_data = load_dataset_multi(n_traces = n_traces,only_t = training_type == 'multi_t',only_s = training_type == 'multi_s',dataset = 'training',permutations = permutations, known_alpha = known_alpha) 
        elif 'hierarchical' in training_type:           
            X_profiling , validation_data = load_dataset_hierarchical(n_traces = n_traces,dataset = 'training') 
        else:
            X_profiling , validation_data = load_dataset(byte,n_traces = n_traces,t = 'single_t' in training_type , alpha_known = 'first' in training_type,dataset = 'training') 
        X_profiling = X_profiling.shuffle(len(X_profiling)).batch(batch_size) 
        validation_data = validation_data.batch(batch_size)
        history = model.fit(X_profiling, batch_size=batch_size, verbose = 1, epochs=epochs[cycle], validation_data=validation_data,callbacks =callbacks)
    print('Saved model ! ')
 
    file = open(METRICS_FOLDER+'history_training_'+(model_t ),'wb')
    pickle.dump(history.history,file)
    file.close()

    
    
    return model






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--MULTI_MODEL',   action="store_true", dest="MULTI_MODEL", help='Adding the masks to the labels', default=False)
    parser.add_argument('--SHARED',   action="store_true", dest="SHARED", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI',   action="store_true", dest="MULTI", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI_S',   action="store_true", dest="MULTI_S", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI_T',   action="store_true", dest="MULTI_T", help='Adding the masks to the labels', default=False)
    parser.add_argument('--SINGLE_S',   action="store_true", dest="SINGLE_S", help='Adding the masks to the labels', default=False)
    parser.add_argument('--SINGLE_T',   action="store_true", dest="SINGLE_T", help='Adding the masks to the labels', default=False)
    parser.add_argument('--FIRST',   action="store_true", dest="FIRST", help='Adding the masks to the labels', default=False)
    parser.add_argument('--HIERARCHICAL',   action="store_true", dest="HIERARCHICAL", help='Adding the masks to the labels', default=False)                                 
    args            = parser.parse_args()
  
    MULTI_MODEL        = args.MULTI_MODEL

    SHARED        = args.SHARED
    MULTI = args.MULTI
    MULTI_S = args.MULTI_S
    MULTI_T= args.MULTI_T
    SINGLE_T=  args.SINGLE_T
    SINGLE_S = args.SINGLE_S
    FIRST = args.FIRST
    HIERARCHICAL = args.HIERARCHICAL

    if MULTI:
        TRAINING_TYPE = 'multi'
    elif HIERARCHICAL:
        TRAINING_TYPE = 'hierarchical'
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
    if 'multi' in TRAINING_TYPE or 'hierarchical' in TRAINING_TYPE:
        for SHARED in [True,False]:
            process_eval = Process(target=train_model, args=(MULTI_MODEL,SHARED,TRAINING_TYPE,'all'))
            process_eval.start()
            process_eval.join() 
    else:
        for TRAINING_TYPE in ['single_s','single_t']:
            for byte in range(16):
                process_eval = Process(target=train_model, args=(MULTI_MODEL,False,TRAINING_TYPE + ('' if not FIRST else '_first'),byte))
                process_eval.start()
                process_eval.join()                                     


    print("$ Done !")
            
        
        
