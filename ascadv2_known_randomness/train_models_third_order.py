import argparse
import os
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Softmax, Dense, Input,Conv1D, AveragePooling1D, BatchNormalization,Multiply, Concatenate , Conv2D , AveragePooling2D, Reshape, Lambda
from tensorflow.keras.optimizers import Adam , RMSprop

from multiprocessing import Process


# import dataset paths and variables
from utility import  METRICS_FOLDER , MODEL_FOLDER

# import custom layers
from utility import MultiLayer , XorLayer , PoolingCrop , SharedWeightsDenseLayer

from utility import load_dataset_third_order , load_dataset_multi_third_order 
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


########################## FULLY EXTRACTED SCENARIO #################################################



def model_single_task_xor_extracted(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units, learning_rate=0.0001, classes=256 , name ='',summary = True):
    inputs_dict = {}
    outputs = {}
    
    inputs_intermediate  = Input(shape = (93,1) ,name = 'inputs_intermediate')
    inputs_rin = Input(shape = (1605,1) ,name = 'inputs_rin')
    inputs_alpha = Input(shape = (4088,1) ,name = 'inputs_alpha')
    
    inputs_dict['inputs_intermediate'] = inputs_intermediate   
    inputs_dict['inputs_rin'] = inputs_rin   
    inputs_dict['inputs_alpha'] = inputs_alpha   
    
    branch_rin = cnn_core(inputs_rin,convolution_blocks = 1, kernel_size = [32],filters = 16, strides = 5, pooling_size = 2)
    branch_rin = dense_core(branch_rin,dense_blocks = 2,dense_units = 256,activated = True,name = 'rin')
    
    branch_alpha = cnn_core(inputs_alpha,convolution_blocks = 1, kernel_size = [64],filters = 16, strides = 5, pooling_size = 4)
    branch_alpha = dense_core(branch_alpha,dense_blocks = 2,dense_units = 256,activated = True,name = 'alpha')    
    
    branch_intermediate = cnn_core(inputs_intermediate,convolution_blocks = convolution_blocks, kernel_size = kernel_size,filters = filters, strides = 1, pooling_size = pooling_size)
    intermediate_branch = dense_core(branch_intermediate,dense_blocks = dense_blocks,dense_units = dense_units,activated = False)
    output_before_alpha = XorLayer(name ='output_before_alpha' )([intermediate_branch,branch_rin]) 
    
    output = MultiLayer(name = 'multi')([output_before_alpha,branch_alpha])
    outputs['output'] = Softmax(name = 'output')(output)
    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_single_task_xor_{}'.format(name))

 
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if summary:
        model.summary() 
    return model


def model_multi_task_extracted(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units, learning_rate=0.0001, classes=256 , name ='',summary = True):
    
    inputs_dict = {}
    
    inputs_intermediate  = Input(shape = (93,16,1) ,name = 'inputs_intermediate')
    inputs_rin = Input(shape = (1605,1) ,name = 'inputs_rin')
    inputs_alpha = Input(shape = (4088,1) ,name = 'inputs_alpha')
    
    inputs_dict['inputs_intermediate'] = inputs_intermediate   
    inputs_dict['inputs_rin'] = inputs_rin   
    inputs_dict['inputs_alpha'] = inputs_alpha   
    
    branch_rin = cnn_core(inputs_rin,convolution_blocks = 1, kernel_size = [32],filters = 16, strides = 5, pooling_size = 2)
    branch_rin = dense_core(branch_rin,dense_blocks = 2,dense_units = 256,activated = True,name = 'rin')
    
    branch_alpha = cnn_core(inputs_alpha,convolution_blocks = 1, kernel_size = [64],filters = 16, strides = 5, pooling_size = 4)
    branch_alpha = dense_core(branch_alpha,dense_blocks = 2,dense_units = 256,activated = True,name = 'alpha')    
    

    outputs = {} 
   
    
    for byte in range(16):
        branch_intermediate = cnn_core(inputs_intermediate[:,:,byte],convolution_blocks = convolution_blocks, kernel_size = kernel_size, strides = 1,filters = filters, pooling_size = pooling_size)
        intermediate_branch = dense_core(branch_intermediate,dense_blocks = dense_blocks,dense_units = dense_units,activated = False)
        output_before_alpha = XorLayer(name ='output_before_alpha_{}'.format(byte) )([intermediate_branch,branch_rin]) 
        output = MultiLayer(name = 'multi_t_{}'.format(byte))([output_before_alpha,branch_alpha])        
        outputs['output_t_{}'.format(byte)] = Softmax(name = 'output_t_{}'.format(byte))(output)
          
    losses = {}   
    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'])
    if summary:
        model.summary()
    return model   



def model_multi_task_noperm(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units, learning_rate=0.0001, classes=256 , name ='',summary = True):
    
    inputs_dict = {}
    
    inputs_intermediate  = Input(shape = (93,16,1) ,name = 'inputs_intermediate')
    inputs_permutations  = Input(shape = (93,16,1) ,name = 'inputs_permutations')
    inputs_rin = Input(shape = (1605,1) ,name = 'inputs_rin')
    inputs_alpha = Input(shape = (4088,1) ,name = 'inputs_alpha')
    
    inputs_dict['inputs_intermediate'] = inputs_intermediate   
    inputs_dict['inputs_permutations'] = inputs_permutations  
    inputs_dict['inputs_rin'] = inputs_rin   
    inputs_dict['inputs_alpha'] = inputs_alpha   
    
    branch_rin = cnn_core(inputs_rin,convolution_blocks = 1, kernel_size = [32],filters = 16, strides = 5, pooling_size = 2)
    branch_rin = dense_core(branch_rin,dense_blocks = 2,dense_units = 256,activated = True,name = 'rin')
    
    branch_alpha = cnn_core(inputs_alpha,convolution_blocks = 1, kernel_size = [64],filters = 16, strides = 5, pooling_size = 4)
    branch_alpha = dense_core(branch_alpha,dense_blocks = 2,dense_units = 256,activated = True,name = 'alpha')    
    
    branch_inputs = mlp_core(inputs_intermediate[:,:,:,0],3,93)
    branch_permutations = mlp_core(inputs_permutations[:,:,:,0],3,93)

    inputs_byte  = Concatenate(axis = 1)([branch_inputs,branch_permutations])
    intermediate_branch = dense_core_shared(inputs_byte,dense_blocks = 1,dense_units = 2048,activated = False)  

    outputs = {} 
   
    branches_permutations = []
    branches_intermediates = []
    for byte in range(16):
        
        # cbranch_intermediate = cnn_core(inputs_byte[:,:,byte],convolution_blocks = convolution_blocks, kernel_size = kernel_size, strides = 1,filters = filters, pooling_size = pooling_size)
        
        # branch_permutations = dense_core(branch_permutations_all[:,:,byte],dense_blocks = 2,dense_units = 256,activated = True,permutations = True,name = 'permutations_{}'.format(byte))    
        # outputs['p_{}'.format(byte)] = branch_permutations
        # branches_permutations.append(tf.expand_dims(branch_permutations,2))
        
        output_before_alpha = XorLayer(name ='output_before_alpha_{}'.format(byte) )([intermediate_branch[:,:,byte],branch_rin]) 
        output = MultiLayer(name = 'multi_t_{}'.format(byte))([output_before_alpha,branch_alpha])    
        outputs['output_t_{}'.format(byte)] = Softmax(name = 'output_t_{}'.format(byte))(output)
     #     branches_intermediates.append(tf.expand_dims(Softmax()(output),2))
    
    # branches_permutations = Concatenate(axis = 2)(branches_permutations)
    # branches_intermediates = Concatenate(axis = 2)(branches_intermediates)
    
    # branches = tf.matmul(branches_intermediates,branches_permutations)
    
    # for byte in range(16):
    #     outputs['output_t_{}'.format(byte)] =Lambda(lambda x: x, name = "output_t_{}".format(byte))(branches[:,:,byte]) 
    
    
    
    
    losses = {}   
    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'])
    if summary:
        model.summary()
    return model   








########################## SEPARATED SCENARIO #################################################









def model_single_task_xor_flat(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units, learning_rate=0.0001, classes=256 , name ='',summary = True):
    inputs_dict = {}
    
    inputs_intermediate  = Input(shape = (93 * 16,1) ,name = 'inputs_intermediate')
    inputs_rin = Input(shape = (1605,1) ,name = 'inputs_rin')
    inputs_alpha = Input(shape = (4088,1) ,name = 'alpha')
    inputs_dict['inputs_intermediate'] = inputs_intermediate   
    inputs_dict['inputs_rin'] = inputs_rin   
    inputs_dict['inputs_alpha'] = inputs_alpha   


    branch_rin = cnn_core(inputs_rin,convolution_blocks = 1, kernel_size = [32],filters = 16, strides = 5, pooling_size = 2)
    branch_rin = dense_core(branch_rin,dense_blocks = 2,dense_units = 256,activated = True,name = 'rin')
    
    branch_alpha = cnn_core(inputs_alpha,convolution_blocks = 1, kernel_size = [64],filters = 16, strides = 5, pooling_size = 4)
    branch_alpha = dense_core(branch_alpha,dense_blocks = 2,dense_units = 256,activated = True,name = 'alpha')    
    

    outputs = {} 
    branch_intermediate = cnn_core(inputs_intermediate,convolution_blocks = convolution_blocks, kernel_size = kernel_size, strides = 1,filters = filters, pooling_size = pooling_size)   
    intermediate_branch = dense_core(branch_intermediate,dense_blocks = dense_blocks,dense_units = dense_units,activated = False)
    
    output_before_alpha = XorLayer(name ='output_before_alpha' )([intermediate_branch,branch_rin])    
    output = MultiLayer(name = 'multi')([output_before_alpha,branch_alpha])
    outputs['output'] = Softmax(name = 'output')(output)
    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_single_task_xor_{}'.format(name))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if summary:
        model.summary() 
    return model


### Multi-Task Models









def model_multi_task_flat(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units, learning_rate=0.0001, classes=256 , name ='',summary = True):
    
    inputs_dict = {}
    
    inputs_intermediate  = Input(shape = (93 * 16,1) ,name = 'inputs_intermediate')
    inputs_rin = Input(shape = (1605,1) ,name = 'inputs_rin')
    inputs_alpha = Input(shape = (4088,1) ,name = 'inputs_alpha')
    inputs_dict['inputs_intermediate'] = inputs_intermediate   
    inputs_dict['inputs_rin'] = inputs_rin   
    inputs_dict['inputs_alpha'] = inputs_alpha   
    
    branch_rin = cnn_core(inputs_rin,convolution_blocks = 1, kernel_size = [32],filters = 16, strides = 5, pooling_size = 2)
    branch_rin = dense_core(branch_rin,dense_blocks = 2,dense_units = 256,activated = True,name = 'rin')
    
    branch_alpha = cnn_core(inputs_alpha,convolution_blocks = 1, kernel_size = [64],filters = 16, strides = 5, pooling_size = 4)
    branch_alpha = dense_core(branch_alpha,dense_blocks = 2,dense_units = 256,activated = True,name = 'alpha')    
    

    outputs = {} 
    branch_intermediate = cnn_core(inputs_intermediate,convolution_blocks = convolution_blocks, kernel_size = kernel_size, strides = 1,filters = filters, pooling_size = pooling_size)   
    
    for byte in range(16):

        intermediate_branch = dense_core(branch_intermediate,dense_blocks = dense_blocks,dense_units = dense_units,activated = False)
        output_before_alpha = XorLayer(name ='output_before_alpha_{}'.format(byte) )([intermediate_branch,branch_rin]) 
        output = MultiLayer(name = 'multi_t_{}'.format(byte))([output_before_alpha,branch_alpha])        
        outputs['output_t_{}'.format(byte)] = Softmax(name = 'output_t_{}'.format(byte))(output)
        
    losses = {}   
    
    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'])
    if summary:
        model.summary()
    return model   



######################## CONCATENATED SCENARIO ################################


def model_single_task_xor_whole(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units, learning_rate=0.0001, classes=256 , name ='',summary = True):
    inputs_dict = {}
    
    traces  = Input(shape = (93 * 16 + 1605+4088,1) ,name = 'traces')

    inputs_dict['traces'] = traces   
   

    branch_rin = cnn_core(traces,convolution_blocks = 1, kernel_size = [32],filters = 16, strides = 5, pooling_size = 2)
    branch_rin = dense_core(branch_rin,dense_blocks = 2,dense_units = 256,activated = True,name = 'rin')
    
    branch_alpha = cnn_core(traces,convolution_blocks = 1, kernel_size = [64],filters = 16, strides = 5, pooling_size = 4)
    branch_alpha = dense_core(branch_alpha,dense_blocks = 2,dense_units = 256,activated = True,name = 'alpha')    
    

    outputs = {} 
    branch_intermediate = cnn_core(traces,convolution_blocks = convolution_blocks, kernel_size = kernel_size, strides = 1,filters = filters, pooling_size = pooling_size)   
    intermediate_branch = dense_core(branch_intermediate,dense_blocks = dense_blocks,dense_units = dense_units,activated = False)
    
    output_before_alpha = XorLayer(name ='output_before_alpha' )([intermediate_branch,branch_rin])    
    output = MultiLayer(name = 'multi')([output_before_alpha,branch_alpha])
    outputs['output'] = Softmax(name = 'output')(output)
    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_single_task_xor_{}'.format(name))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if summary:
        model.summary() 
    return model


def model_multi_task_whole(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units, learning_rate=0.0001, classes=256 , name ='',summary = True):
    
    inputs_dict = {}
    
    traces  = Input(shape = (93 * 16 + 1605 + 4088,1) ,name = 'traces')
    inputs_dict['traces'] = traces     
      
    
    branch_rin = cnn_core(traces,convolution_blocks = 1, kernel_size = [32],filters = 16, strides = 5, pooling_size = 2)
    branch_rin = dense_core(branch_rin,dense_blocks = 2,dense_units = 256,activated = True,name = 'rin')
    
    branch_alpha = cnn_core(traces,convolution_blocks = 1, kernel_size = [64],filters = 16, strides = 5, pooling_size = 4)
    branch_alpha = dense_core(branch_alpha,dense_blocks = 2,dense_units = 256,activated = True,name = 'alpha')    
    

    outputs = {} 
    branch_intermediate = cnn_core(traces,convolution_blocks = convolution_blocks, kernel_size = kernel_size, strides = 1,filters = filters, pooling_size = pooling_size)   
    
    for byte in range(16):

        intermediate_branch = dense_core(branch_intermediate,dense_blocks = dense_blocks,dense_units = dense_units,activated = False)
        output_before_alpha = XorLayer(name ='output_before_alpha_{}'.format(byte) )([intermediate_branch,branch_rin]) 
        output = MultiLayer(name = 'multi_t_{}'.format(byte))([output_before_alpha,branch_alpha])        
        outputs['output_t_{}'.format(byte)] = Softmax(name = 'output_t_{}'.format(byte))(output)
    losses = {}   

    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'])
    if summary:
        model.summary()
    return model   


######################## ARCHITECTURE BUILDING ################################

def cnn_core(inputs_core,convolution_blocks , kernel_size,filters, strides , pooling_size):
    x = inputs_core
    for block in range(convolution_blocks):
        x = Conv1D(kernel_size=(kernel_size[block],), strides=strides, filters=filters, activation='selu', padding='same',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)    
        x = BatchNormalization()(x)
        x = AveragePooling1D(pool_size = pooling_size)(x)
    
    output_layer = Flatten()(x) 

    return output_layer

def cnn_core_2D(inputs_core,convolution_blocks , kernel_size,filters, strides , pooling_size):
    x = inputs_core
    for block in range(convolution_blocks):
        x = Conv2D(kernel_size=(kernel_size[block],16), strides=(1,1), filters=filters, activation='selu', padding='same',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)    
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size = (pooling_size,1))(x)
    
    x = tf.transpose(x, [0,1,3,2])
    output_layer = Reshape((-1,16))(x)
    
    

    return output_layer

def mlp_core(input_core,dense_blocks,dense_units):
    x = input_core
    
    for block in range(dense_blocks):
       
        x = SharedWeightsDenseLayer(input_dim = x.shape[1],units = dense_units//(2*(block + 1)))(x)        
        x = BatchNormalization(axis = 1)(x)
    return x
def dense_core(inputs_core,dense_blocks,dense_units,permutations = False,activated = False,name = ''):
    x = inputs_core
    
    for block in range(dense_blocks):
        x = Dense(dense_units, activation='selu',kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)        
        x = BatchNormalization()(x)
        
    if activated:
        output_layer = Dense(256 if not permutations else 16,activation ='softmax' ,kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed),name = 'output_{}'.format(name))(x)  
    else:
        output_layer = Dense(256 if not permutations else 16,kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed))(x)   
    return output_layer    

def dense_core_shared(inputs_core,dense_blocks,dense_units,permutations = False,activated = False,name = ''):
    x = inputs_core
    
    for block in range(dense_blocks):
        print(x.shape)
        x = SharedWeightsDenseLayer(input_dim = x.shape[1],units = dense_units)(x)        
        x = BatchNormalization(axis = 1)(x)
        
    output_layer = SharedWeightsDenseLayer(input_dim = x.shape[1],units = 256,activation = False)(x)   
    return output_layer 
#### Training high level function

def train_model(training_type,byte,convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units):
    epochs = 250
    batch_size = 250
    n_traces = 325000
    

    if  'single_task' in training_type:
        flat = 'flat' in training_type
        whole = 'whole' in training_type
        X_profiling , validation_data = load_dataset_third_order(byte,flat = flat,whole = whole,n_traces = n_traces,dataset = 'training')
        model_t = 'model_{}'.format(training_type)
    else:
        
        flat = training_type == 'multi_task_flat'
        whole = training_type == 'multi_task_whole'
        noperm = training_type == 'multi_task_noperm'
        X_profiling , validation_data = load_dataset_multi_third_order(flat = flat,noperm = noperm,whole = whole,n_traces = n_traces,dataset = 'training') 
        model_t = 'model_{}'.format(training_type)
    
    all_callbacks = []
    monitor = 'val_accuracy'
    mode = 'max'
    strides = None
    if model_t == 'model_single_task_extracted':
        model = model_single_task_extracted(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units)       
    elif model_t == 'model_single_task_xor_extracted':
        model = model_single_task_xor_extracted(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units)
    elif model_t == 'model_single_task_flat':
        model = model_single_task_flat(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units)            
    elif model_t == 'model_single_task_xor_flat':
        model = model_single_task_xor_flat(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units)         
    elif model_t == 'model_single_task_whole':
        model = model_single_task_whole(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units)  
    elif model_t == 'model_single_task_xor_whole':
        model = model_single_task_xor_whole(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units)     
    elif model_t == 'model_multi_task_extracted':
        model = model_multi_task_extracted(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units)                  
        monitor = 'val_min_accuracy'   
        mode = 'max'    
        all_callbacks.append(CombinedMetric())
    elif model_t == 'model_multi_task_noperm':
        model = model_multi_task_noperm(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units)                  
        monitor = 'val_min_accuracy'   
        mode = 'max'    
        all_callbacks.append(CombinedMetric())

    elif model_t == 'model_multi_task_whole':
        model = model_multi_task_whole(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units)                  
        monitor = 'val_min_accuracy'   
        mode = 'max'    
        all_callbacks.append(CombinedMetric())
    elif model_t == 'model_multi_task_flat':
        model = model_multi_task_flat(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units)                  
        monitor = 'val_min_accuracy'   
        mode = 'max'    
        all_callbacks.append(CombinedMetric())
    else:
        print('Some error here')

    
    X_profiling = X_profiling.shuffle(len(X_profiling)).batch(batch_size) 
    validation_data = validation_data.batch(batch_size)
    id_model = 'cb{}ks{}f{}{}ps{}db{}du{}'.format(convolution_blocks , kernel_size,filters,'' if strides is None else 's{}'.format(strides), pooling_size,dense_blocks,dense_units)
    file_name = '{}_{}_{}_third'.format(model_t,byte if not ('model_multi_task' in model_t) else 'all',id_model) 
    callbacks = tf.keras.callbacks.ModelCheckpoint(
                                filepath= MODEL_FOLDER+ file_name+'.h5',
                                save_weights_only=True,
                                monitor=monitor,
                                mode=mode,
                                save_best_only=True)

    all_callbacks.append(callbacks)

    
    history = model.fit(X_profiling, batch_size=batch_size, verbose = 1, epochs=epochs, validation_data=validation_data,callbacks =all_callbacks)
    print('Saved model {} ! '.format(file_name))
 
    file = open(METRICS_FOLDER+'history_training_'+(file_name ),'wb')
    pickle.dump(history.history,file)
    file.close()

    
    
    return model






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--SINGLE_TASK',   action="store_true", dest="SINGLE_TASK", help='Adding the masks to the labels', default=False)
    parser.add_argument('--SINGLE_TASK_XOR',   action="store_true", dest="SINGLE_TASK_XOR", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI',   action="store_true", dest="MULTI", help='Adding the masks to the labels', default=False)
    parser.add_argument('--SINGLE_TASK_SOFTMAX_CHECK',   action="store_true", dest="SINGLE_TASK_SOFTMAX_CHECK", help='Adding the masks to the labels', default=False)
    parser.add_argument('--ALL',   action="store_true", dest="ALL", help='Adding the masks to the labels', default=False)
        
    args            = parser.parse_args()
  
    SINGLE_TASK        = args.SINGLE_TASK

    SINGLE_TASK_XOR        = args.SINGLE_TASK_XOR
    MULTI = args.MULTI
    ALL = args.ALL
    SINGLE_TASK_SOFTMAX_CHECK= args.SINGLE_TASK_SOFTMAX_CHECK

    TARGETS = {}
    if SINGLE_TASK:
        training_types = ['single_task_extracted', 'single_task_flat','single_task_whole']
    elif SINGLE_TASK_SOFTMAX_CHECK:
        training_types = ['single_task_softmax_check']

    elif SINGLE_TASK_XOR:
        training_types = ['single_task_xor_flat']

    elif MULTI:
        training_types = ['multi_task_noperm']
    elif ALL:
        training_types = ['single_task_extracted', 'single_task_flat','single_task_whole','single_task_softmax_check','single_task_xor_extracted', 'single_task_xor_flat','single_task_xor_whole','multi_task_extracted','multi_task_flat','multi_task_whole']
    else:
        print('No training mode selected')
    # for model_random in tqdm(range(25)):
    #     convolution_blocks = 1
    #     kernel_size = sorted(np.random.randint(4,32,size = convolution_blocks))       
    #     filters = 16
    #     pooling_size = np.random.randint(2,5)
    #     dense_blocks = np.random.randint(1,5)
    #     dense_units = np.random.randint(64,512)
    #     print(model_random,kernel_size,pooling_size,dense_blocks,dense_units) 
        

    for model_random in tqdm(range(0,25)):
    
        convolution_blocks = 1
        kernel_size = sorted(np.random.randint(4,32,size = convolution_blocks))       
        filters = 16
        pooling_size = np.random.randint(2,5)
        dense_blocks = np.random.randint(1,5)
        dense_units = np.random.randint(64,512)
        if model_random in  [22]:
            for training_type in training_types:
                print(training_type)
                if  not ('multi_task' in training_type):
                    for byte in range(0,1):
                        process_eval = Process(target=train_model, args=(training_type,byte,convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units))
                        process_eval.start()
                        process_eval.join()
                else:
                    process_eval = Process(target=train_model, args=(training_type,'all',convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units))
                    process_eval.start()
                    process_eval.join()                                    


    print("$ Done !")
            
        
        
