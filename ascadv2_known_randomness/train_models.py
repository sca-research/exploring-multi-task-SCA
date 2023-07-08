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


def model_hierarchical( learning_rate=0.001, classes=256 , name ='',summary = True):
    
    inputs_dict = {}
    outputs = {}
    
    inputs_block  = Input(shape = (93,16) ,name = 'inputs_block')
    inputs_dict['inputs_block'] = inputs_block 
    block_core = SharedWeightsDenseLayer(input_dim = inputs_block.shape[1],units = 64,shares = 16)(inputs_block)      
    block_core = BatchNormalization(axis = 1)(block_core)
    
    inputs_permutations  = Input(shape = (93,16) ,name = 'inputs_permutations')
    inputs_dict['inputs_permutations'] = inputs_permutations  
    permutations_core = SharedWeightsDenseLayer(input_dim = inputs_permutations.shape[1],units = 64,shares = 16)(inputs_permutations)      
    permutations_core = BatchNormalization(axis = 1)(permutations_core)
    
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
    
    # inputs_mj = Input(shape = (25,16) ,name = 'inputs_mj')
    # inputs_dict['inputs_mj'] = inputs_mj  
    # mj_core = SharedWeightsDenseLayer(input_dim = inputs_mj.shape[1],units = 16,shares = 16)(inputs_mj)      
    # mj_core = BatchNormalization(axis = 1)(mj_core)
    # mj_core = dense_core_shared(mj_core,output_units = 256)
    
    
    # inputs_m = Input(shape = (24* 4,4) ,name = 'inputs_m')
    # inputs_dict['inputs_m'] = inputs_m  
    # m_core = SharedWeightsDenseLayer(input_dim = inputs_m.shape[1],shares = 4,units = 32)(inputs_m)      
    # m_core = BatchNormalization(axis = 1)(m_core)   
    
    # input_to_output =[]
    # for byte in range(0,16,4):
    #     branch = byte // 4
    #     m_core_branch = dense_core_shared(m_core[:,:,branch],branches = 4,output_units = 8)
    #     concat = Concatenate(axis = 1)([m_core_branch[:,:,0],m_core_branch[:,:,1],m_core_branch[:,:,2],m_core_branch[:,:,3]])
    #     outputs['m_sig_{}'.format(branch)] =  Activation('sigmoid',name = 'branch_{}'.format(branch))(concat)
    #     input_to_output.append(m_core_branch)
    # m_core = Concatenate(axis = 2)(input_to_output)
    # m_core = dense_core_shared(m_core,non_shared_block = 0, shared_block = 0,branches = 16, output_units = 256,split = True)
    # for byte in range(16):
    #     outputs['output_m_{}'.format(byte)] = Softmax(name = 'output_m_{}'.format(byte))(m_core[:,:,byte])
        
    
    # inputs_s_mj = Input(shape = (10,16) ,name = 'inputs_s_mj')
    # inputs_dict['inputs_s_mj'] = inputs_s_mj  
    # s_mj_core = SharedWeightsDenseLayer(input_dim = inputs_s_mj.shape[1],units = 4,shares = 16)(inputs_s_mj)      
    # s_mj_core = BatchNormalization(axis = 1)(s_mj_core)
    # s_mj_core = dense_core_shared(s_mj_core,output_units = 256)
    
    # inputs_t_mj = Input(shape = (10,16) ,name = 'inputs_t_mj')    
    # inputs_dict['inputs_t_mj'] = inputs_t_mj 
    # t_mj_core = SharedWeightsDenseLayer(input_dim = inputs_t_mj.shape[1],units = 4,shares = 16)(inputs_t_mj)      
    # t_mj_core = BatchNormalization(axis = 1)(t_mj_core)
    # t_mj_core = dense_core_shared(t_mj_core,output_units = 256)
    
    

    s_beta_core = dense_core_shared(block_core,output_units = 256)
    # s_beta_mj_core = dense_core_shared(block_core,output_units = 256)
    t_rin_core = dense_core_shared(block_core,output_units = 256)
    # t_rin_mj_core = dense_core_shared(block_core,output_units = 256)   
    
    permutations_core = dense_core_shared(permutations_core,output_units = 16) 
    permutations_matrix = []
    for byte in range(16):
        outputs['s_beta_{}'.format(byte)] = Softmax(name = 'output_s_beta_{}'.format(byte))(s_beta_core[:,:,byte])
        # outputs['output_s_beta_mj_{}'.format(byte)] =  Softmax(name = 'output_s_beta_mj_{}'.format(byte),dtype = tf.float64)(s_beta_mj_core[:,:,byte])
        outputs['t_rin_{}'.format(byte)] =  Softmax(name = 'output_t_rin_{}'.format(byte))(t_rin_core[:,:,byte])
        # outputs['output_t_rin_mj_{}'.format(byte)] =  Softmax(name = 'output_t_rin_mj_{}'.format(byte),dtype = tf.float64)(t_rin_mj_core[:,:,byte])
        # outputs['output_mj_{}'.format(byte)] =  Softmax(name = 'output_mj_{}'.format(byte))(mj_core[:,:,byte])

        outputs['j_{}'.format(byte)] =  Softmax(name = 'output_j_{}'.format(byte))(permutations_core[:,:,byte])
        permutations_matrix.append(tf.expand_dims(outputs['j_{}'.format(byte)],2))
    permutations_matrix = Concatenate(axis = 2)(permutations_matrix)
    # mj_from_m_all = tf.matmul(m_core,tf.transpose(permutations_matrix,[0,2,1]))
    metrics = {}
    tj_matrix = []
    sj_matrix = []
    for byte in range(16):   

        # mj = Add()([mj_from_m_all[:,:,byte]  , mj_core[:,:,byte] ])
        # # mj = Softmax(name = 'output_mj_{}'.format(byte))(mj)
        # mj = Softmax()(mj)
        
        # beta_mj = XorLayer(name = 'xor_beta_mj_{}'.format(byte))([mj,outputs['output_beta']])
        # rin_mj = XorLayer(name = 'xor_rin_mj_{}'.format(byte))([mj,outputs['output_rin']])

        # xor_sj_mj =  XorLayer(name = 'xor_sj_mj_{}'.format(byte))([s_mj_core[:,:,byte],mj])
        # xor_sj_beta_mj =  XorLayer(name = 'xor_sj_beta_mj_{}'.format(byte))([s_beta_mj_core[:,:,byte],beta_mj])
        xor_sj_beta =  XorLayer(name = 'xor_sj_beta_{}'.format(byte))([s_beta_core[:,:,byte],outputs['beta']])
        # alpha_sj = Add()([xor_sj_mj,xor_sj_beta_mj,xor_sj_beta])
        
        sj = MultiLayer(name = 'multi_s_{}'.format(byte))([xor_sj_beta,outputs['alpha']])
        outputs['sj_{}'.format(byte)] = Softmax(name = 'output_sj_{}'.format(byte))(sj)     
        sj = InvSboxLayer(name = 'inv_s_{}'.format(byte))(sj)
        
        
        
        # xor_tj_mj =  XorLayer(name = 'xor_tj_mj_{}'.format(byte))([t_mj_core[:,:,byte],mj])
        # xor_tj_rin_mj =  XorLayer(name = 'xor_tj_rin_mj_{}'.format(byte))([t_rin_mj_core[:,:,byte],rin_mj])
        xor_tj_rin =  XorLayer(name = 'xor_tj_rin_{}'.format(byte))([t_rin_core[:,:,byte],outputs['rin']])
        # alpha_tj = Add()([xor_tj_mj,xor_tj_rin_mj,xor_tj_rin])
        tj = MultiLayer(name = 'multi_t_{}'.format(byte))([xor_tj_rin,outputs['alpha']])
        add = Add_Shares(shares = 2, name = 'add_shares_{}'.format(byte))([tf.keras.activations.selu(sj),tf.keras.activations.selu(tj)])
        outputs['tj_{}'.format(byte)] = Softmax(name = 'output_tj_{}'.format(byte))(add)     
              
        
        tj_matrix.append(tf.expand_dims(add,2))
    tj_matrix = Concatenate(axis = 2)(tj_matrix)
    

    
    t_from_tj = tf.matmul(tj_matrix,tf.transpose(permutations_matrix,[0,2,1]))
    
        
    for byte in range(16): 
        
        outputs['output_t_{}'.format(byte)] = Softmax(name = 'output_t_{}'.format(byte))(t_from_tj[:,:,byte])     
        metrics['output_t_{}'.format(byte)] ='accuracy'

          
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


def dense_core(inputs_core,dense_blocks,dense_units,activated = False,name = ''):
    x = inputs_core
    
    for block in range(dense_blocks):
        x = Dense(dense_units, activation='selu')(x)        
        # x = BatchNormalization()(x)
        
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

def train_model(model_type):
    epochs =25
    batch_size = 500
    n_traces = 450000
    
    model_t = 'model_{}'.format(model_type)   
    if model_type == 'hierarchical': 
        model , losses , metrics , weights  = model_hierarchical()
    elif model_type == 'sbox_input':       
        model , losses , metrics , weights  = model_sbox_input()       
    elif model_type == 'sbox_output':       
        model , losses , metrics , weights  = model_sbox_output()  
    elif model_type == 'permutations':       
        model , losses , metrics , weights  = model_permutations()  
    else:
        print('You fucked up')
    




    learning_rates = [0.0001,0.001,0.00001]
    for cycle in range(3):
        callbacks = tf.keras.callbacks.ModelCheckpoint(
                                    filepath= MODEL_FOLDER+ model_t+'{}.h5'.format(cycle),
                                    save_weights_only=True,
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=True)
        optimizer = Adam(learning_rate=learning_rates[cycle])
        model.compile(loss=losses, optimizer=optimizer, metrics=metrics , loss_weights = weights)
        X_profiling , validation_data = load_dataset_multi(n_traces = n_traces,dataset = 'training',model_type = model_type) 
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

    args            = parser.parse_args()
  
    HIERARCHICAL        = args.HIERARCHICAL

    FLAT        = args.FLAT


    TARGETS = {}
    model_types = []
    if HIERARCHICAL:
        model_types  = ['hierarchical']
    elif FLAT:
        model_types = ['permutations']

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
        process_eval = Process(target=train_model, args=(model_type,))
        process_eval.start()
        process_eval.join()                                    


    print("$ Done !")
            
        
        
