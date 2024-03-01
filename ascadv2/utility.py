import os
import pickle
import h5py


import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import tensorflow.experimental.numpy as tnp

tnp.experimental_enable_numpy_behavior()

from utils.generate_intermediate_values import multGF256

from gmpy2 import f_divmod_2exp

# Opening dataset specific variables 

file = open('utils/dataset_parameters','rb')
parameters = pickle.load(file)
file.close()


DATASET_FOLDER  = parameters['DATASET_FOLDER']
METRICS_FOLDER = DATASET_FOLDER + 'metrics/' 
MODEL_FOLDER = DATASET_FOLDER + 'models/' 
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


class XorLayer(tf.keras.layers.Layer):
  def __init__(self,classes =256 ,name = ''):
    super(XorLayer, self).__init__(name = name)
    all_maps = np.zeros((classes,classes),dtype =np.uint8)
    for i in range(classes):
        for j in range(classes):
            all_maps[i, j ] = i^j    
    self.mapping2 = all_maps
    self.classes = classes
    
  def call(self, inputs):  
 
    pred1 = inputs[0]
    pred2 = tnp.asarray(inputs[1])
    
    p1 = pred1 
    p2 = pred2[:,self.mapping2]

    res = tf.einsum('ij,ijk->ik',p1,p2)
    return res
class SharedWeightsDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim=1,units = 1, shares = 16,name = '',activation = True,precision = 'float32',seed = 42):
        if name == '':
            name = 'SharedWeightsDenseLayer_'+str(np.random.randint(0,high = 99999))
        super(SharedWeightsDenseLayer, self).__init__(name = name )
        self.w = self.add_weight(shape=(input_dim,units), dtype=precision,
                                  trainable=True,name = 'weights'+name,initializer=tf.keras.initializers.RandomUniform(seed=seed)
                                  
        )
        print(seed)
        self.b = self.add_weight(shape=(units,shares), dtype=precision,
                                  trainable=True,name = 'biais'+name,initializer=tf.keras.initializers.RandomUniform(seed=seed)
                                  
        )
        self.input_dim = input_dim
        self.shares = shares
        self.activation = activation
        self.precision = tf.float64 if precision == 'float64' else tf.float32

    
        
    def call(self, inputs):
        x = tf.einsum('ijk, jf -> ifk',tf.cast(inputs,self.precision),self.w)    
        if self.activation:
            return tf.keras.activations.selu( x + self.b)
        else:
            return  x + self.b   
log_table=[ 0, 0, 25, 1, 50, 2, 26, 198, 75, 199, 27, 104, 51, 238, 223, 3,
    100, 4, 224, 14, 52, 141, 129, 239, 76, 113, 8, 200, 248, 105, 28, 193,
    125, 194, 29, 181, 249, 185, 39, 106, 77, 228, 166, 114, 154, 201, 9, 120,
    101, 47, 138, 5, 33, 15, 225, 36, 18, 240, 130, 69, 53, 147, 218, 142,
    150, 143, 219, 189, 54, 208, 206, 148, 19, 92, 210, 241, 64, 70, 131, 56,
    102, 221, 253, 48, 191, 6, 139, 98, 179, 37, 226, 152, 34, 136, 145, 16,
    126, 110, 72, 195, 163, 182, 30, 66, 58, 107, 40, 84, 250, 133, 61, 186,
    43, 121, 10, 21, 155, 159, 94, 202, 78, 212, 172, 229, 243, 115, 167, 87,
    175, 88, 168, 80, 244, 234, 214, 116, 79, 174, 233, 213, 231, 230, 173, 232,
    44, 215, 117, 122, 235, 22, 11, 245, 89, 203, 95, 176, 156, 169, 81, 160,
    127, 12, 246, 111, 23, 196, 73, 236, 216, 67, 31, 45, 164, 118, 123, 183,
    204, 187, 62, 90, 251, 96, 177, 134, 59, 82, 161, 108, 170, 85, 41, 157,
    151, 178, 135, 144, 97, 190, 220, 252, 188, 149, 207, 205, 55, 63, 91, 209,
    83, 57, 132, 60, 65, 162, 109, 71, 20, 42, 158, 93, 86, 242, 211, 171,
    68, 17, 146, 217, 35, 32, 46, 137, 180, 124, 184, 38, 119, 153, 227, 165,
    103, 74, 237, 222, 197, 49, 254, 24, 13, 99, 140, 128, 192, 247, 112, 7 ]

alog_table =[1, 3, 5, 15, 17, 51, 85, 255, 26, 46, 114, 150, 161, 248, 19, 53,
    95, 225, 56, 72, 216, 115, 149, 164, 247, 2, 6, 10, 30, 34, 102, 170,
    229, 52, 92, 228, 55, 89, 235, 38, 106, 190, 217, 112, 144, 171, 230, 49,
    83, 245, 4, 12, 20, 60, 68, 204, 79, 209, 104, 184, 211, 110, 178, 205,
    76, 212, 103, 169, 224, 59, 77, 215, 98, 166, 241, 8, 24, 40, 120, 136,
    131, 158, 185, 208, 107, 189, 220, 127, 129, 152, 179, 206, 73, 219, 118, 154,
    181, 196, 87, 249, 16, 48, 80, 240, 11, 29, 39, 105, 187, 214, 97, 163,
    254, 25, 43, 125, 135, 146, 173, 236, 47, 113, 147, 174, 233, 32, 96, 160,
    251, 22, 58, 78, 210, 109, 183, 194, 93, 231, 50, 86, 250, 21, 63, 65,
    195, 94, 226, 61, 71, 201, 64, 192, 91, 237, 44, 116, 156, 191, 218, 117,
    159, 186, 213, 100, 172, 239, 42, 126, 130, 157, 188, 223, 122, 142, 137, 128,
    155, 182, 193, 88, 232, 35, 101, 175, 234, 37, 111, 177, 200, 67, 197, 84,
    252, 31, 33, 99, 165, 244, 7, 9, 27, 45, 119, 153, 176, 203, 70, 202,
    69, 207, 74, 222, 121, 139, 134, 145, 168, 227, 62, 66, 198, 81, 243, 14,
    18, 54, 90, 238, 41, 123, 141, 140, 143, 138, 133, 148, 167, 242, 13, 23,
    57, 75, 221, 124, 132, 151, 162, 253, 28, 36, 108, 180, 199, 82, 246, 1 ]

# Multiplication function in GF(2^8)
def multGF256(a,b):
    if (a==0) or (b==0):
        return 0
    else:
        return alog_table[(log_table[a]+log_table[b]) %255]
    
class MultiLayer(tf.keras.layers.Layer):
    def __init__(self,classes = 256 ,name = ''):
        super(MultiLayer, self).__init__(name = name)
        all_maps = np.load('utils/mult_mapping.npy')
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
        config = {'mapping':self.mapping,
                  'classes':self.classes}
        base_config = super(MultiLayer,self).get_config()
        base_config.update(config)
        return base_config



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
    model_file  = MODEL_FOLDER  + name
    print('Loading model {}'.format(model_file))
    structure.load_weights(model_file)
    return structure   



def adapt_plaintexts(p,k,fake_key):
    new_p = np.empty(p.shape,dtype = np.uint8)
    for trace  in range(len(p)):
        plaintext = int.from_bytes(p[trace],byteorder ='big')
        key = int.from_bytes(k[trace],byteorder ='big')        
        pxork = plaintext ^ key
        new_p[trace] = bytearray((pxork ^ fake_key).to_bytes(16,byteorder ='big'))
        assert int.from_bytes(new_p[trace],byteorder ='big') ^ fake_key ^ plaintext == key
    return new_p


def read_from_h5_file(n_traces = 1000,dataset = 'training',load_plaintexts = False):   
    
    f = h5py.File(DATASET_FOLDER + FILE_DATASET,'r')[dataset]  
    print(FILE_DATASET)
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








    

def load_dataset(byte,t = False,alpha_known = False,n_traces = None,dataset = 'training',encoded_labels = True,print_logs = True):    
    training = dataset == 'training' 
    if print_logs :
        str_targets = 'Loading samples and labels in order to train the multi-task model'
        print(str_targets)
        
    traces , labels_dict = read_from_h5_file(n_traces=n_traces,dataset = dataset)
   
    traces = np.expand_dims(traces,2)

    X_profiling_dict = {}  
    if not alpha_known:
        X_profiling_dict['inputs_alpha'] = traces[:,:2000]
    else:
        X_profiling_dict['alpha'] = get_hot_encode(np.array(labels_dict['alpha'],dtype = np.uint8)[:n_traces])
    if t:
        X_profiling_dict['inputs_rin'] = traces[:,2000:2000+1000]
    else:
        X_profiling_dict['inputs_beta'] = traces[:,3000:3000+200]

    X_profiling_dict['inputs_block'] = traces[:,3200:3200 + 93 * 16].reshape((n_traces,93,16))

    Y_profiling_dict = {}

    s1 = np.array(labels_dict['s1'],dtype = np.uint8)[:n_traces]
    t1 = np.array(labels_dict['t1'],dtype = np.uint8)[:n_traces]
    all_permutations = np.array(labels_dict['p'],dtype = np.uint8)[:n_traces]

    if t:
        Y_profiling_dict['output'] = get_hot_encode( np.array([t1[i,all_permutations[i,byte]] for i in range(n_traces)],dtype = np.uint8))
    else:
        Y_profiling_dict['output'] = get_hot_encode( np.array([s1[i,all_permutations[i,byte]] for i in range(n_traces)],dtype = np.uint8))


    print('Finished loading dataset {}'.format(dataset))
    if training:
        
        traces_val , labels_dict_val = read_from_h5_file(n_traces=n_traces//10,dataset = 'validation')
        traces_val = np.expand_dims(traces_val,2)
        X_validation_dict = {}  
        if not alpha_known: 
            X_validation_dict['inputs_alpha'] = traces_val[:,:2000]
        else:
            X_validation_dict['alpha'] = get_hot_encode(np.array(labels_dict_val['alpha'],dtype =np.uint8)[:n_traces//10])
        if t:
            X_validation_dict['inputs_rin'] = traces_val[:,2000:2000+1000]
        else:
            X_validation_dict['inputs_beta'] = traces_val[:,3000:3000+200]

        X_validation_dict['inputs_block'] = traces_val[:,3200:3200 + 93 * 16].reshape((n_traces//10,93,16))

        Y_validation_dict = {}

        s1 = np.array(labels_dict_val['s1'],dtype = np.uint8)[:n_traces//10]
        t1 = np.array(labels_dict_val['t1'],dtype = np.uint8)[:n_traces//10]
        all_permutations = np.array(labels_dict_val['p'],dtype = np.uint8)[:n_traces//10]

        if t:
            Y_validation_dict['output'] = get_hot_encode( np.array([t1[i,all_permutations[i,byte]] for i in range(n_traces//10)],dtype = np.uint8))
        else:
            Y_validation_dict['output'] = get_hot_encode( np.array([s1[i,all_permutations[i,byte]] for i in range(n_traces//10)],dtype = np.uint8))


        print('Finished loading dataset {}'.format('validation'))

        return tf.data.Dataset.from_tensor_slices((X_profiling_dict ,Y_profiling_dict)), tf.data.Dataset.from_tensor_slices(( X_validation_dict,Y_validation_dict)) 
   
    else:
        return (X_profiling_dict,Y_profiling_dict)   
       


    

def load_dataset_multi(n_traces = None,only_t = False,only_s = False,known_alpha = False,whole = False,dataset = 'training',encoded_labels = True,print_logs = True,permutations = False):
    training = dataset == 'training' 
    if print_logs :
        str_targets = 'Loading samples and labels in order to train the multi-task model'
        print(str_targets)
        
    traces , labels_dict = read_from_h5_file(n_traces=n_traces,dataset = dataset)
   
    traces = np.expand_dims(traces,2)

    X_profiling_dict = {}  
    if not known_alpha:
        X_profiling_dict['inputs_alpha'] = traces[:,:2000]
    else:
        X_profiling_dict['alpha'] = get_hot_encode(np.array(labels_dict['alpha'],dtype = np.uint8)[:n_traces])
    if not only_s:
        X_profiling_dict['inputs_rin'] = traces[:,2000:2000+1000]
    if not only_t:
        X_profiling_dict['inputs_beta'] = traces[:,3000:3000+200]

    X_profiling_dict['inputs_block'] = traces[:,3200:3200 + 93 * 16].reshape((n_traces,93,16))
    Y_profiling_dict = {}

    s1 = np.array(labels_dict['s1'],dtype = np.uint8)[:n_traces]
    t1 = np.array(labels_dict['t1'],dtype = np.uint8)[:n_traces]
    all_permutations = np.array(labels_dict['p'],dtype = np.uint8)[:n_traces]
    for byte in range(16):    
        if only_s:
            Y_profiling_dict['output_{}'.format(byte)] = get_hot_encode( np.array([s1[i,all_permutations[i,byte]] for i in range(n_traces)],dtype = np.uint8))
        elif only_t:
            Y_profiling_dict['output_{}'.format(byte)] = get_hot_encode( np.array([t1[i,all_permutations[i,byte]] for i in range(n_traces)],dtype = np.uint8))
        else:
            Y_profiling_dict['output_tj_{}'.format(byte)] = get_hot_encode( np.array([t1[i,all_permutations[i,byte]] for i in range(n_traces)],dtype = np.uint8))
            Y_profiling_dict['output_sj_{}'.format(byte)] = get_hot_encode( np.array([s1[i,all_permutations[i,byte]] for i in range(n_traces)],dtype = np.uint8))



    print('Finished loading dataset {}'.format(dataset))
    if training:
        
        traces_val , labels_dict_val = read_from_h5_file(n_traces=n_traces//10,dataset = 'validation')
        traces_val = np.expand_dims(traces_val,2)
        X_validation_dict = {}  
        if not known_alpha: 
            X_validation_dict['inputs_alpha'] = traces_val[:,:2000]
        else:
            X_validation_dict['alpha'] = get_hot_encode(np.array(labels_dict_val['alpha'],dtype =np.uint8)[:n_traces//10])
        if not only_s:
            X_validation_dict['inputs_rin'] = traces_val[:,2000:2000+1000]
        if not only_t:
            X_validation_dict['inputs_beta'] = traces_val[:,3000:3000+200]

        X_validation_dict['inputs_block'] = traces_val[:,3200:3200 + 93 * 16].reshape((n_traces//10,93,16))

        Y_validation_dict = {}

        s1 = np.array(labels_dict_val['s1'],dtype = np.uint8)[:n_traces//10]
        t1 = np.array(labels_dict_val['t1'],dtype = np.uint8)[:n_traces//10]
        all_permutations = np.array(labels_dict_val['p'],dtype = np.uint8)[:n_traces//10]
        for byte in range(16):    

            if only_s:
                Y_validation_dict['output_{}'.format(byte)] = get_hot_encode( np.array([s1[i,all_permutations[i,byte]] for i in range(n_traces//10)],dtype = np.uint8))
            elif only_t:
                Y_validation_dict['output_{}'.format(byte)] = get_hot_encode( np.array([t1[i,all_permutations[i,byte]] for i in range(n_traces//10)],dtype = np.uint8))
            else:
                Y_validation_dict['output_tj_{}'.format(byte)] = get_hot_encode( np.array([t1[i,all_permutations[i,byte]] for i in range(n_traces//10)],dtype = np.uint8))
                Y_validation_dict['output_sj_{}'.format(byte)] = get_hot_encode( np.array([s1[i,all_permutations[i,byte]] for i in range(n_traces//10)],dtype = np.uint8))
                

        print('Finished loading dataset {}'.format('validation'))
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
    return tf.one_hot(label_set,depth = classes,dtype = tf.uint8)


