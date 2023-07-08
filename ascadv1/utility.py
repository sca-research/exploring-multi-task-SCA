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



def to_matrix(text):

    matrix = []
    for i in range(4):
        matrix.append([0,0,0,0])
    for i in range(len(text)):
        elem = text[i]
        matrix[i%4][i//4] =elem
    return matrix    








    

def load_dataset(byte, target = 's1',n_traces = None,dataset = 'training',encoded_labels = True,print_logs = True):    
    
    training = dataset == 'training' 
    if print_logs :
        str_targets = 'Loading samples and labels for {}'.format(target)
        print(str_targets)
        
    traces , labels_dict  = read_from_h5_file(n_traces=n_traces,dataset = dataset)     
    traces = np.expand_dims(traces,2)
    X_profiling_dict = {}  
    X_profiling_dict['traces'] = traces

    if training:
        traces_val , labels_dict_val = read_from_h5_file(n_traces=10000,dataset = 'test')
        traces_val = np.expand_dims(traces_val,2)
        X_validation_dict = {}  
        X_validation_dict['traces'] = traces_val


    Y_profiling_dict = {}
   
    real_values_t1 = np.array(labels_dict[target],dtype = np.uint8)[:n_traces]
    Y_profiling_dict['output'] = get_hot_encode(real_values_t1[:,byte],classes = 256) if encoded_labels else  real_values_t1 
   
    if training:
        Y_validation_dict = {}

        real_values_t1_val = np.array(labels_dict_val[target],dtype = np.uint8)[:10000]
        Y_validation_dict['output'] = get_hot_encode(real_values_t1_val[:,byte],classes = 256 )   
        return tf.data.Dataset.from_tensor_slices((X_profiling_dict ,Y_profiling_dict)), tf.data.Dataset.from_tensor_slices(( X_validation_dict,Y_validation_dict)) 
   
    else:
        return (X_profiling_dict,Y_profiling_dict)  
       


def load_dataset_multi(target,n_traces = None,dataset = 'training',encoded_labels = True,print_logs = True):
    training = dataset == 'training' 
    if print_logs :
        str_targets = 'Loading samples and labels in order to train the multi-task model'
        print(str_targets)
        
    traces , labels_dict = read_from_h5_file(n_traces=n_traces,dataset = dataset)
    traces = np.expand_dims(traces,2)
    
    
    
    X_profiling_dict = {}  
    X_profiling_dict['traces'] = traces

    if training:
        traces_val , labels_dict_val = read_from_h5_file(n_traces=10000,dataset = 'test')
        traces_val = np.expand_dims(traces_val,2)
        
        X_validation_dict = {}  
        X_validation_dict['traces'] = traces_val






    if print_logs :
        print('Loaded inputs')    
        

    Y_profiling_dict = {}
    if not target == 'k1':
        real_values_temp = np.array(labels_dict[target],dtype = np.uint8)[:n_traces]
        for byte in range(2,16):
            Y_profiling_dict['output_{}'.format(byte)] = get_hot_encode(real_values_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_temp[:,byte] 
    else:
        real_values_s1_temp = np.array(labels_dict['s1'],dtype = np.uint8)[:n_traces]
        real_values_t1_temp = np.array(labels_dict['t1'],dtype = np.uint8)[:n_traces]
        for byte in range(2,16):
            Y_profiling_dict['output_s_{}'.format(byte)] = get_hot_encode(real_values_s1_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_s1_temp[:,byte] 
            Y_profiling_dict['output_t_{}'.format(byte)] = get_hot_encode(real_values_t1_temp[:,byte],classes = 256 ) if encoded_labels else  real_values_t1_temp[:,byte] 
                                              

                                           
    if training:
        Y_validation_dict = {}
        if not target == 'k1':
            real_values_temp_val = np.array(labels_dict_val[target],dtype = np.uint8)[:10000]
            for byte in range(2,16):
                Y_validation_dict['output_{}'.format(byte)] = get_hot_encode(real_values_temp_val[:,byte],classes = 256 )   
        else:
            real_values_s1_temp_val = np.array(labels_dict_val['s1'],dtype = np.uint8)[:10000]
            real_values_t1_temp_val = np.array(labels_dict_val['t1'],dtype = np.uint8)[:10000]
            for byte in range(2,16):
                Y_validation_dict['output_s_{}'.format(byte)] = get_hot_encode(real_values_s1_temp_val[:,byte],classes = 256 )   
                Y_validation_dict['output_t_{}'.format(byte)] = get_hot_encode(real_values_t1_temp_val[:,byte],classes = 256 )   




        return tf.data.Dataset.from_tensor_slices((X_profiling_dict ,Y_profiling_dict)), tf.data.Dataset.from_tensor_slices(( X_validation_dict,Y_validation_dict)) 
   
    else:
        return (X_profiling_dict,Y_profiling_dict)    


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

