import os
import pickle
import h5py


import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import tensorflow.experimental.numpy as tnp

from tensorflow.keras.layers import BatchNormalization

tnp.experimental_enable_numpy_behavior()


from gmpy2 import f_divmod_2exp

# Opening dataset specific variables 

file = open('utils/dataset_parameters','rb')
parameters = pickle.load(file)
file.close()


DATASET_FOLDER  = parameters['DATASET_FOLDER']
METRICS_FOLDER = DATASET_FOLDER + 'metrics_exp/' 
MODEL_FOLDER = DATASET_FOLDER + 'models_exp/' 
TRACES_FOLDER = DATASET_FOLDER + 'traces/'
REALVALUES_FOLDER = DATASET_FOLDER + 'real_values/'
POWERVALUES_FOLDER = DATASET_FOLDER + 'powervalues/'
TIMEPOINTS_FOLDER = DATASET_FOLDER + 'timepoints/'
KEY_FIXED = parameters['KEY']
FILE_DATASET = parameters['FILE_DATASET']
MASKS = parameters['MASKS']
ONE_MASKS = parameters['ONE_MASKS']
INTERMEDIATES = parameters['INTERMEDIATES']
VARIABLE_LIST = parameters['VARIABLE_LIST']
MASK_INTERMEDIATES = parameters['MASK_INTERMEDIATES']
MASKED_INTERMEDIATES = parameters['MASKED_INTERMEDIATES']




shift_rows_s = list([
    0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11
    ])






class PoolingCrop(tf.keras.layers.Layer):
    def __init__(self, input_dim=1, use_dropout = True,name = ''):
        if name == '':
            name = 'Crop_'+str(np.random.randint(0,high = 99999))
        super(PoolingCrop, self).__init__(name = name )
        self.w = self.add_weight(shape=(input_dim,1), dtype="float32",
                                  trainable=True,name = 'weights'+name
                                  
        )
        self.input_dim = input_dim
        self.pooling = tf.keras.layers.AveragePooling1D(pool_size = 2,strides = 2,padding = 'same')
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = tf.keras.layers.AlphaDropout(0.01)
        self.bn = BatchNormalization()
        
        
    
        
    def call(self, inputs):
        kernel = tf.multiply(self.w, inputs)       
        pooling = self.pooling(kernel)
        output = self.bn(pooling)
        if self.use_dropout:
            output = self.dropout(output)
        return output
    
    def get_config(self):
        config = {'w':self.w,
                  'input_dim' : self.input_dim,
                  'pooling' : self.pooling,
                  'dropout' : self.dropout
                  }
        base_config = super(PoolingCrop,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    

class InvSboxLayer(tf.keras.layers.Layer):
  def __init__(self,name = ''):
    super(InvSboxLayer, self).__init__(name = name)
    self.mapping = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

  def call(self, inputs):  
    pred = tnp.asarray(inputs)[:,self.mapping]
    return tf.convert_to_tensor(pred)

    def get_config(self):
        config = {'mapping':self.mapping}
        base_config = super(InvSboxLayer,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SharedWeightsDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim=1,units = 1, shares = 16,name = '',activation = True):
        if name == '':
            name = 'SharedWeightsDenseLayer_'+str(np.random.randint(0,high = 99999))
        super(SharedWeightsDenseLayer, self).__init__(name = name )
        self.w = self.add_weight(shape=(input_dim,units), dtype="float32",
                                  trainable=True,name = 'weights'+name,initializer=tf.keras.initializers.RandomUniform(seed=7)
                                  
        )
        self.b = self.add_weight(shape=(units,shares), dtype="float32",
                                  trainable=True,name = 'biais'+name,initializer=tf.keras.initializers.RandomUniform(seed=7)
                                  
        )
        self.input_dim = input_dim
        self.shares = shares
        self.activation = activation

    
        
    def call(self, inputs):
        x = tf.einsum('ijk, jf -> ifk',inputs,self.w)    
        if self.activation:
            return tf.keras.activations.selu( x + self.b)
        else:
            return  x + self.b    

class XorLayer(tf.keras.layers.Layer):
  def __init__(self,classes =256 ,name = ''):
    super(XorLayer, self).__init__(name = name)
    all_maps = np.load('utils/xor_mapping.npy')
    mapping1 = []
    mapping2 = []
    for classe in range(classes):
        mapped = np.where(all_maps[classe] == 1)
        mapping1.append(mapped[0])
        mapping2.append(mapped[1])
    self.mapping1 = np.array(mapping1)
    self.mapping2 = np.array(mapping2)
    self.classes = classes
    
  def call(self, inputs):  
 
    pred1 = tnp.asarray(inputs[0])
    pred2 = tnp.asarray(inputs[1])
    p1 = pred1[:,self.mapping1]
    p2 = pred2[:,self.mapping2]

    res = tf.reduce_sum(tf.multiply(p1,p2),axis =2)   
    return res

    def get_config(self):
        config = {'mapping':self.mapping}
        base_config = super(XorLayer,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

def get_pow_rank(x):
    if x == 2 :
        return 1
    if x == 1 :
        return 0
    n = 1
    q  , r = f_divmod_2exp(x , n)    
    while q > 0:
        q  , r = f_divmod_2exp(x , n)
        n += 1
    return n

def get_rank(result,true_value):
    key_probabilities_sorted = np.argsort(result)[::-1]
    key_ranking_good_key = list(key_probabilities_sorted).index(true_value) + 1
    return key_ranking_good_key


def load_model_from_name(structure , name):
    model_file  = MODEL_FOLDER + name
    print('Loading model {}'.format(model_file))
    structure.load_weights(model_file)
    return structure   




def read_from_h5_file(n_traces = 1000,dataset = 'training',load_plaintexts = False):   
    
    f = h5py.File(DATASET_FOLDER + FILE_DATASET,'r')[dataset]  
    labels_dict = f['labels']
    if load_plaintexts:
        data =  {'keys':f['keys']   ,'plaintexts':f['plaintexts']}
        return  f['traces'][:n_traces] , labels_dict, data
    else:
        return  f['traces'][:n_traces] , labels_dict

def get_byte(i):
    for b in range(17,1,-1):
        if str(b) in i:
            return b
    return -1
    


def to_matrix(text):

    matrix = []
    for i in range(4):
        matrix.append([0,0,0,0])
    for i in range(len(text)):
        elem = text[i]
        matrix[i%4][i//4] =elem
    return matrix    








    

def load_dataset(target,byte,n_traces = None,dataset = 'training',encoded_labels = True,print_logs = True):    
   
    training = dataset == 'training' 
    if print_logs :
        str_targets = 'Loading samples and labels for {}'.format(target)
        print(str_targets)
        
    traces , labels_dict  = read_from_h5_file(n_traces=n_traces,dataset = dataset)     
    traces = np.expand_dims(traces,2)
    X_profiling_dict = {}  
    X_profiling_dict['traces'] = traces

    if training:
        traces_val , labels_dict_val = read_from_h5_file(n_traces=n_traces//10,dataset = 'test')
        traces_val = np.expand_dims(traces_val,2)
        X_validation_dict = {}  
        X_validation_dict['traces'] = traces_val


    Y_profiling_dict = {}
   
    real_values = np.array(labels_dict[target],dtype = np.uint8)[:n_traces]
    Y_profiling_dict['output'] = get_hot_encode(real_values[:,byte],classes = 256) if encoded_labels else  real_values
   
    if training:
        Y_validation_dict = {}

        real_values_val = np.array(labels_dict_val[target],dtype = np.uint8)[:n_traces//10]
        Y_validation_dict['output'] = get_hot_encode(real_values_val[:,byte],classes = 256 )   
        return tf.data.Dataset.from_tensor_slices((X_profiling_dict ,Y_profiling_dict)), tf.data.Dataset.from_tensor_slices(( X_validation_dict,Y_validation_dict)) 
   
    else:
        return (X_profiling_dict,Y_profiling_dict)  
       

def normalise_neural_trace(v):
    # Shift up
    return v - np.min(v)

def normalise_neural_trace_single(v):
    return divide_rows_by_max(normalise_neural_trace(v))

def divide_rows_by_max(X):
    if len(X.shape) == 1:
        return X / np.max(X)
    else:
        return X / np.max(X, axis=1)[:, None]
def normalise_neural_traces(X):
    divided_by_max = divide_rows_by_max(X)
    return divided_by_max

def normalise_traces_to_int8(x):
    x = normalise_neural_traces(x)
    x = x * 128
    return x.astype(np.int8)

def load_dataset_multi(target,n_traces = None,dataset = 'training',encoded_labels = True,print_logs = True):
    training = dataset == 'training' 
    if print_logs :
        str_targets = 'Loading samples and labels in order to train the multi-task model'
        print(str_targets)
        
    traces , labels_dict  = read_from_h5_file(n_traces=n_traces,dataset = dataset)     
    traces = np.expand_dims(traces,2)
    X_profiling_dict = {}  
    X_profiling_dict['traces'] = traces

    if training:
        traces_val , labels_dict_val = read_from_h5_file(n_traces=n_traces//10,dataset = 'test')
        traces_val = np.expand_dims(traces_val,2)
        X_validation_dict = {}  
        X_validation_dict['traces'] = traces_val


    if print_logs :
        print('Loaded inputs')    
        

    Y_profiling_dict = {}

        
    real_values_temp = np.array(labels_dict['t1'],dtype = np.uint8)[:n_traces]        
    real_values_i = np.array(labels_dict['i'],dtype = np.uint8)[:n_traces]      
    Y_profiling_dict['output_i'] = get_hot_encode(real_values_i,classes = 256 ) if encoded_labels else real_values_i
    real_values_t_i_temp = np.array(labels_dict['t1^i'],dtype = np.uint8)[:n_traces]      
    real_values_s_r_temp = np.array(labels_dict['s1^r'],dtype = np.uint8)[:n_traces]   
    real_values_t_ri_temp = np.array(labels_dict['t1^ri'],dtype = np.uint8)[:n_traces] 
    real_values_t_r_temp = np.array(labels_dict['t1^r'],dtype = np.uint8)[:n_traces]    
    real_values_r_temp = np.array(labels_dict['r'],dtype = np.uint8)[:n_traces] 
    for byte in range(2,16):
        Y_profiling_dict['output_{}'.format(byte)] = get_hot_encode(real_values_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_temp[:,byte] 
        Y_profiling_dict['output_t_i_{}'.format(byte)] = get_hot_encode(real_values_t_i_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_t_i_temp[:,byte] 
        Y_profiling_dict['output_s_r_{}'.format(byte)] = get_hot_encode(real_values_s_r_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_s_r_temp[:,byte]
        Y_profiling_dict['output_t_r_{}'.format(byte)] = get_hot_encode(real_values_t_r_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_t_r_temp[:,byte] 
        Y_profiling_dict['output_t_ri_{}'.format(byte)] = get_hot_encode(real_values_t_ri_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_t_ri_temp[:,byte] 
        Y_profiling_dict['output_r_{}'.format(byte)] = get_hot_encode(real_values_r_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_r_temp[:,byte] 

                                           
    if training:
        Y_validation_dict = {}
   
        real_values_temp = np.array(labels_dict_val['t1'],dtype = np.uint8)[:n_traces//10]        
        real_values_i = np.array(labels_dict_val['i'],dtype = np.uint8)[:n_traces//10]      
        Y_validation_dict['output_i'] = get_hot_encode(real_values_i,classes = 256 ) if encoded_labels else real_values_i
        real_values_t_i_temp = np.array(labels_dict_val['t1^i'],dtype = np.uint8)[:n_traces//10]      
        real_values_s_r_temp = np.array(labels_dict_val['s1^r'],dtype = np.uint8)[:n_traces//10]   
        real_values_t_ri_temp = np.array(labels_dict_val['t1^ri'],dtype = np.uint8)[:n_traces//10] 
        real_values_t_r_temp = np.array(labels_dict_val['t1^r'],dtype = np.uint8)[:n_traces//10]    
        real_values_r_temp = np.array(labels_dict_val['r'],dtype = np.uint8)[:n_traces//10] 
        for byte in range(2,16):
            Y_validation_dict['output_{}'.format(byte)] = get_hot_encode(real_values_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_temp[:,byte] 
            Y_validation_dict['output_t_i_{}'.format(byte)] = get_hot_encode(real_values_t_i_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_t_i_temp[:,byte] 
            Y_validation_dict['output_s_r_{}'.format(byte)] = get_hot_encode(real_values_s_r_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_s_r_temp[:,byte] 
            Y_validation_dict['output_t_r_{}'.format(byte)] = get_hot_encode(real_values_t_r_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_t_r_temp[:,byte] 
            Y_validation_dict['output_t_ri_{}'.format(byte)] = get_hot_encode(real_values_t_ri_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_t_ri_temp[:,byte] 
            Y_validation_dict['output_r_{}'.format(byte)] = get_hot_encode(real_values_r_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_r_temp[:,byte] 



        return tf.data.Dataset.from_tensor_slices((X_profiling_dict ,Y_profiling_dict)), tf.data.Dataset.from_tensor_slices(( X_validation_dict,Y_validation_dict)) 
   
    else:
        return (X_profiling_dict,Y_profiling_dict)    





def get_hw(k):
    hw = 0
    for _ in range(8):
        hw += k & 1
        k = k >> 1
    return hw 

def convert_to_binary(e):
    return [1 if e & (1 << (7-n)) else 0 for n in range(8)]   




def get_rank_list_from_prob_dist(probdist,l):
    res =  []
    accuracy = 0
    size = len(l)
    res_score = []
    accuracy_top5 = 0
    for i in range(size):
        rank = get_rank(probdist[i],l[i])
        res.append(rank)
        res_score.append(probdist[i][l[i]])
        accuracy += 1 if rank == 1 else 0
        accuracy_top5 += 1 if rank <= 5 else 0
    return res,(accuracy/size)*100,res_score , (accuracy_top5/size)*100


def get_hot_encode(label_set,classes = 256):    
    return np.eye(classes)[label_set]

