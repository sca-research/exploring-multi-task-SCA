from utility import read_from_h5_file , adapt_plaintexts , get_hot_encode , load_model_from_name , get_rank , get_pow_rank
from utility import XorLayer , InvSboxLayer, MultiLayer
from utility import METRICS_FOLDER , MODEL_FOLDER
from gmpy2 import mpz,mul

from train_models import model_hierarchical , model_permutations , model_sbox_output , model_sbox_input

import argparse , parse
from multiprocessing import Process
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import pickle 
import os
import tensorflow_probability as tfp

class Add_Shares(tf.keras.layers.Layer):
    def __init__(self, input_dim=256,units = 256,shares = 1,name = ''):
        super(Add_Shares, self).__init__(name = name )
        
        self.w = self.add_weight(shape=(input_dim,), dtype="float32",trainable=True, name ='weights')
        
        self.shares = shares
        self.input_dim = input_dim


        
        
    def call(self, inputs):  
        exp_w = tf.math.exp(self.w)
        sum_exp_w = tf.reduce_sum(exp_w)
        w = exp_w/sum_exp_w
        

        x1 = tf.math.multiply(inputs[0], w) 
        x2 = tf.math.multiply(inputs[1],- w) 
        self.add_loss(tf.reduce_sum((x1 + x2)**2))

        return tf.math.log( tf.math.divide(inputs[0],w) +tf.math.divide(inputs[1], w))



class Attack:
    def __init__(self,n_experiments = 1000,n_traces = 5000,model_type = 'multi_task'):
        

        self.n_experiments = n_experiments
        self.n_traces = n_traces
        self.powervalues = {}

        traces , labels_dict, metadata  = read_from_h5_file(n_traces = self.n_traces,dataset = 'attack',load_plaintexts = True)


        self.correct_guesses = {}
        self.history_score = {}
        self.traces_per_exp = 5
        self.predictions_hierarchical = np.zeros((16,self.n_traces,256),dtype =np.float32)
        self.predictions_other = np.zeros((16,self.n_traces,256),dtype =np.float32)
        
        plaintexts = np.array(metadata['plaintexts'],dtype = np.uint8)[:self.n_traces]
        keys =  np.array(metadata['keys'],dtype = np.uint8)[:self.n_traces]
        self.key = 0x00112233445566778899AABBCCDDEEFF
        master_key =[0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                      0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF ]  
          

        
        self.plaintexts = tf.cast(get_hot_encode(adapt_plaintexts(plaintexts,keys,self.key)),tf.float32)
        batch_size = self.n_traces//10
  

        
        

        # id_model  = 'cb{}ks{}f{}ps{}db{}du{}'.format(convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units)
      
        
        # multi_name = 'model_hierarchical2.h5'
        # X_multi = {}
        # model_struct , _ , _ , _ = model_hierarchical()
        # self.model = load_model_from_name(model_struct,multi_name)  
        
        # X_multi['inputs_alpha'] = traces[:,:2000]
        # X_multi['inputs_rin'] = traces[:,2000:2000+1000]
        # X_multi['inputs_beta'] = traces[:,3000:3000+200]
        #  # X_profiling_dict['inputs_m'] = traces[:,3200:3200 + 24 * 16].reshape((n_traces,24*4,4))
        #  # X_profiling_dict['inputs_mj'] = traces[:,3584:3584 + 25 * 16].reshape((n_traces,25,16))
        #  # X_profiling_dict['inputs_s_mj'] = traces[:,3984:3984 + 10 * 16].reshape((n_traces,10,16))
        #  # X_profiling_dict['inputs_t_mj'] = traces[:,4144:4144 + 10 * 16].reshape((n_traces,10,16))
        # X_multi['inputs_block'] = traces[:,4304:4304 + 93 * 16].reshape((n_traces,93,16))
        # X_multi['inputs_permutations'] = traces[:,4304+ 93 * 16:4304+ 93 * 16 + 93 * 16].reshape((n_traces,93,16))
        # predictions = self.model.predict(X_multi,batch_size  = 200)
        

        # # array_predictions_p = np.empty((16,n_traces,16),dtype = np.float32)
        # # predictions_non_permuted = np.empty((16,n_traces,256),dtype = np.float32)
        # # for byte in range(16):
        # #     array_predictions_p[byte] = predictions['j_{}'.format(byte)]
        # #     predictions_non_permuted[byte] = predictions['tj_{}'.format(byte)]

 
            

        # for batch in tqdm(range(self.n_traces// batch_size)):
        #     for byte in range(16):                   
        
        #         self.predictions_hierarchical[byte][batch_size*batch:batch_size*(batch +1)] = XorLayer()([predictions['output_t_{}'.format(byte)][batch_size*batch:batch_size*(batch +1)],self.plaintexts[batch_size*batch:batch_size*(batch +1),byte]])
        
        # else:
            
            
        X_perm = {}
        X_so = {}
        X_si = {}
        model_struct_so , _ , _ , _ = model_sbox_output()
        self.model_sbox_output = load_model_from_name(model_struct_so,'model_sbox_output2.h5')  
        model_struct_si , _ , _ , _ = model_sbox_input()
        self.model_sbox_input = load_model_from_name(model_struct_si,'model_sbox_input2.h5')  
        model_perm , _ , _ , _ = model_permutations()
        self.model_perm = load_model_from_name(model_perm,'model_permutations2.h5')         
        
        X_so['inputs_alpha'] = traces[:,:2000]
        X_si['inputs_alpha'] = traces[:,:2000]
        X_so['inputs_beta'] = traces[:,3000:3000+200]
        X_si['inputs_rin'] = traces[:,2000:2000+1000]
         # X_profiling_dict['inputs_m'] = traces[:,3200:3200 + 24 * 16].reshape((n_traces,24*4,4))
         # X_profiling_dict['inputs_mj'] = traces[:,3584:3584 + 25 * 16].reshape((n_traces,25,16))
         # X_profiling_dict['inputs_s_mj'] = traces[:,3984:3984 + 10 * 16].reshape((n_traces,10,16))
         # X_profiling_dict['inputs_t_mj'] = traces[:,4144:4144 + 10 * 16].reshape((n_traces,10,16))
        X_so['inputs_block'] = traces[:,4304:4304 + 93 * 16].reshape((n_traces,93,16))
        X_si['inputs_block'] = traces[:,4304:4304 + 93 * 16].reshape((n_traces,93,16))
        X_perm['inputs_permutations'] = traces[:,4304+ 93 * 16:4304+ 93 * 16 + 93 * 16].reshape((n_traces,93,16))
        predictions_sj = self.model_sbox_output.predict(X_so,batch_size  = 200)           
        predictions_tj = self.model_sbox_input.predict(X_si,batch_size  = 200)     
        predictions_p = self.model_perm.predict(X_perm,batch_size  = 200)  
        predictions_non_permuted_s = np.empty((n_traces,256,16),dtype = np.float32)
        predictions_non_permuted_t = np.empty((n_traces,256,16),dtype = np.float32)
        
        array_predictions_p = np.empty((n_traces,16,16),dtype = np.float32)
        xor_layer = XorLayer(name = 'xor_layer')
        multi_layer = MultiLayer(name = 'multi_layer')
        inv_layer = InvSboxLayer(name = 'inv_layer')
        for byte in range(16):
            array_predictions_p[:,byte] = predictions_p['j_{}'.format(byte)]
        
        for batch in tqdm(range(self.n_traces// batch_size)):  
            for byte in range(16):
                a_sj = xor_layer([predictions_sj['output_s_beta_{}'.format(byte)][batch_size*batch:batch_size*(batch +1)],predictions_sj['beta'][batch_size*batch:batch_size*(batch +1)]])
                sj = multi_layer([a_sj,predictions_sj['alpha'][batch_size*batch:batch_size*(batch +1)]])
                tj_from_sj = inv_layer(sj)
                
                a_tj = xor_layer([predictions_tj['output_t_rin_{}'.format(byte)][batch_size*batch:batch_size*(batch +1)],predictions_tj['rin'][batch_size*batch:batch_size*(batch +1)]])
                tj = multi_layer([a_tj,predictions_tj['alpha'][batch_size*batch:batch_size*(batch +1)]])          
                
                predictions_non_permuted_t[batch_size*batch:batch_size*(batch +1),:,byte] = tj
                predictions_non_permuted_s[batch_size*batch:batch_size*(batch +1),:,byte] = tj_from_sj
            
        predictions_s = tf.matmul(predictions_non_permuted_s,array_predictions_p)
        predictions_t = tf.matmul(predictions_non_permuted_t,array_predictions_p)
                     
        for batch in tqdm(range(self.n_traces// batch_size)):
            for byte in range(16):                   
        
                self.predictions_other[byte][batch_size*batch:batch_size*(batch +1)] = XorLayer()([predictions_s[batch_size*batch:batch_size*(batch +1),:,byte],self.plaintexts[batch_size*batch:batch_size*(batch +1),byte]])
                self.predictions_hierarchical[byte][batch_size*batch:batch_size*(batch +1)] = XorLayer()([predictions_t[batch_size*batch:batch_size*(batch +1),:,byte],self.plaintexts[batch_size*batch:batch_size*(batch +1),byte]])
                
      
        master_key = np.array(master_key,dtype = np.int32)
        self.subkeys = master_key
        output = []
        input_hierarchical = tf.keras.layers.Input((16,256))
        input_other = tf.keras.layers.Input((16,256))
        for byte in range(16):
            add = Add_Shares(name = 'output_{}'.format(byte))([input_hierarchical[:,byte],input_other[:,byte]])
            output.append(add)
        inputs_dict = {'hierarchical':input_hierarchical,'other':input_other}
        
        self.model_attack = tf.keras.models.Model(inputs = inputs_dict,outputs = output)
          
        
        
        
        
        # 3, 2, 53, 53, 42, 0, 3, 24, 76, 2, 22, 0, 13, 2, 6, 93]

        


        
    def run(self,print_logs = False):
       history_score = {}
       for experiment in tqdm(range(self.n_experiments)):
           if print_logs:
               print('====================')
               print('Experiment {} '.format(experiment))
           history_score[experiment] = {}
           history_score[experiment]['total_rank'] =  [] 
           subkeys_guess = {}
           for i in range(16):
               subkeys_guess[i] = np.zeros(256,)            
           
               history_score[experiment][i] = []
           traces_order = np.random.permutation(self.n_traces)[:self.traces_per_exp] 
           count_trace = 1
           input_dict = {}
           
           input_dict['hierarchical'] = np.swapaxes(self.predictions_hierarchical[:,self.traces_per_exp * experiment:self.traces_per_exp *(experiment + 1) ],1,0)
           input_dict['other'] = np.swapaxes(self.predictions_other[:,self.traces_per_exp * experiment:self.traces_per_exp *(experiment + 1)],1,0)
           lr = [0.01,0.0001]
           memory = 999999999
           last_value = 1
           cycle= 0
           
           # 5e-10 20 traces [255, 251, 218, 255, 175, 255, 243, 238, 190, 240, 247, 255, 255, 253, 249, 187]
           
           # 5e-10 200 traces [252, 253, 254, 254, 255, 255, 255, 252, 251, 255, 253, 253, 253, 252, 253, 255]

           epsi = 5e-10*self.traces_per_exp
           while last_value+epsi < memory:
               optimizer = tf.keras.optimizers.Adam(learning_rate= lr[cycle%2])
               self.model_attack.compile( optimizer=optimizer)  
               history = self.model_attack.fit(input_dict,epochs = 50,verbose = 1,batch_size = self.traces_per_exp)
               
               
               if cycle % 2 == 0:
                   memory = last_value
                   last_value = history.history['loss'][-1]
                   print(last_value)
               cycle += 1
                   
           
           results = []
           for layer in self.model_attack.layers:
                if 'output' in layer.name:
                    n_byte = int(''.join(layer.name.split('_')[1:]))
                    exp_w = tf.math.exp(layer.weights[0])
                    sum_exp_w = tf.reduce_sum(exp_w)
                    score = exp_w/sum_exp_w
                                  
                   
                    print(score)
                    print(self.subkeys[n_byte])

                    results.append(  np.where(np.argsort(score)== self.subkeys[n_byte])[0][0])
           print(results)
           # predictions = self.model_attack.predict(input_dict)
           # for byte in range(16):
           #     subkeys_guess[byte] += np.sum(predictions[byte])
           #     results.append(  np.where(np.argsort(subkeys_guess[byte])== self.subkeys[byte])[0][0])
           # print(results)
           break
                   
                   
           
           
           
           
           
           
       


                
   
def run_attack(model_type):                
    attack = Attack(model_type= model_type)
    attack.run()
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--FLAT', action="store_true", dest="FLAT",
                        help='Single task models', default=False)

    args            = parser.parse_args()
  

    FLAT        = args.FLAT


    TARGETS = {}



    MODEL_TYPE = ['flat' if FLAT else 'hierarchical']


    for model_type in MODEL_TYPE:

        process_eval = Process(target=run_attack, args=(model_type,)  )
        process_eval.start()
        process_eval.join()
                            
            
            
    
    
            
            
        
        