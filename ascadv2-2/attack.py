from utility import read_from_h5_file , adapt_plaintexts , get_hot_encode , load_model_from_name , get_rank , get_pow_rank
from utility import XorLayer 
from utility import METRICS_FOLDER , MODEL_FOLDER
from gmpy2 import mpz,mul

from train_models import model_hierarchical

import argparse , parse
from multiprocessing import Process
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import pickle 
import os

class Attack:
    def __init__(self,n_experiments = 1000,n_traces = 5000,model_type = 'multi_task'):
        
        self.models = {}
        self.n_experiments = n_experiments
        self.n_traces = n_traces
        self.powervalues = {}

        traces , labels_dict, metadata  = read_from_h5_file(n_traces = self.n_traces,dataset = 'attack',load_plaintexts = True)


        self.correct_guesses = {}
        self.history_score = {}
        self.traces_per_exp = 100
        self.predictions = np.zeros((16,self.n_traces,256),dtype =np.float32)
        
        
        plaintexts = np.array(metadata['plaintexts'],dtype = np.uint8)[:self.n_traces]
        keys =  np.array(metadata['keys'],dtype = np.uint8)[:self.n_traces]
        self.permutations = np.array(labels_dict['p'],dtype = np.uint8)[:self.n_traces]
        self.key = 0x00112233445566778899AABBCCDDEEFF
        master_key =[0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                      0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF ]  
          

        
        self.plaintexts = tf.cast(get_hot_encode(adapt_plaintexts(plaintexts,keys,self.key)),tf.float32)
        batch_size = self.n_traces//10
  

        
        

        # id_model  = 'cb{}ks{}f{}ps{}db{}du{}'.format(convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units)
      
        multi_name = 'model.h5'
        X_multi = {}
        model_struct , _ , _  = model_hierarchical()
        self.models['all'] = load_model_from_name(model_struct,multi_name)  
        
        X_multi['inputs_alpha'] = traces[:,:2000]
        X_multi['inputs_rin'] = traces[:,2000:2000+1000]
        X_multi['inputs_beta'] = traces[:,3000:3000+200]
         # X_profiling_dict['inputs_m'] = traces[:,3200:3200 + 24 * 16].reshape((n_traces,24*4,4))
         # X_profiling_dict['inputs_mj'] = traces[:,3584:3584 + 25 * 16].reshape((n_traces,25,16))
         # X_profiling_dict['inputs_s_mj'] = traces[:,3984:3984 + 10 * 16].reshape((n_traces,10,16))
         # X_profiling_dict['inputs_t_mj'] = traces[:,4144:4144 + 10 * 16].reshape((n_traces,10,16))
        X_multi['inputs_block'] = traces[:,4304:4304 + 93 * 16].reshape((n_traces,93,16))
        # X_multi['inputs_permutations'] = traces[:,4304+ 93 * 16:4304+ 93 * 16 + 93 * 16].reshape((n_traces,93,16))
        predictions = self.models['all'].predict(X_multi,batch_size  = 200)
        permuted_predictions= np.empty((self.n_traces,16,256),dtype = np.float32)
        for elem in range(self.n_traces):
            for byte in range(16):
                permuted_predictions[elem,self.permutations[elem,byte]] = predictions['output_tj_{}'.format(byte)][elem]
           


     
                
        # for batch in tqdm(range(self.n_traces// batch_size)):        
        #     for byte in range(16):
        #         for byte_perm in range(16):
        #             self.predictions[byte_perm][batch_size*batch:batch_size*(batch +1)] = tf.add(self.predictions[byte_perm,batch_size*batch:batch_size*(batch +1)], tf.expand_dims(tf.cast(self.permutations[byte,batch_size*batch:batch_size*(batch +1),byte_perm],tf.float32),1) * predictions_non_permuted[byte,batch_size*batch:batch_size*(batch +1)] ) 
                               
        # for batch in tqdm(range(self.n_traces// batch_size)):
        #     for byte in range(16):                   
        
        #         self.predictions[byte][batch_size*batch:batch_size*(batch +1)] = XorLayer()([self.predictions[byte,batch_size*batch:batch_size*(batch +1)],self.plaintexts[batch_size*batch:batch_size*(batch +1),byte]])
                
   
        for batch in tqdm(range(self.n_traces//batch_size)):
            for byte in range(16):                   
                
                self.predictions[byte][batch_size*batch:batch_size*(batch +1)] = XorLayer()([permuted_predictions[batch_size*batch:batch_size*(batch +1),byte] ,self.plaintexts[batch_size*batch:batch_size*(batch +1),byte]])

       
        master_key = np.array(master_key,dtype = np.int32)
        self.subkeys = master_key
        
        
        


        
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
           
           for trace in traces_order:
               
               
               
               recovered  = {}
               all_recovered = True
               ranks = {}
               if print_logs:
                   print('========= Trace {} ========='.format(count_trace))
               rank_string = ""
               total_rank = mpz(1)
               for byte in range(16):
                   subkeys_guess[byte] += np.log(self.predictions[byte][trace] + 1e-36)
                  
                   ranks[byte] = get_rank(subkeys_guess[byte],self.subkeys[byte])
                   history_score[experiment][byte].append(ranks[byte])
                   total_rank = mul(total_rank,mpz(ranks[byte]))
                   rank_string += "| rank for byte {} : {} | \n".format(byte,ranks[byte])
                   if np.argmax(subkeys_guess[byte]) == self.subkeys[byte]:
                       recovered[byte] = True                        
                   else:
                       recovered[byte] = False
                       all_recovered = False                
              
               history_score[experiment]['total_rank'].append(get_pow_rank(total_rank))
               if print_logs:
                   print(rank_string)
                   print('Total rank 2^{}'.format( history_score[experiment]['total_rank'][-1]))
                   print('\n')
               if all_recovered:  
                   if print_logs:
                       
                       print('All bytes Recovered at trace {}'.format(count_trace))
                   
                   for elem in range(count_trace,self.traces_per_exp):
                       for i in range(16):
                           history_score[experiment][byte].append(ranks[byte])
                       history_score[experiment]['total_rank'].append(1)
                   break
                   count_trace += 1
               else:
                   count_trace += 1
               if print_logs:
                   print('\n')
           if not all_recovered:
                 for fake_experiment in range(self.n_experiments):
                     history_score[fake_experiment] = {}
                     history_score[fake_experiment]['total_rank'] =  [] 
                     for byte in range(16):
                         history_score[fake_experiment][byte] = []
                     for elem in range(self.traces_per_exp):
                         for byte in range(16):
                             history_score[fake_experiment][byte].append(128)
                         history_score[fake_experiment]['total_rank'].append(128)
                 break
       array_total_rank = np.empty((self.n_experiments,self.traces_per_exp))
       for i in range(self.n_experiments):
           for j in range(self.traces_per_exp):
               array_total_rank[i][j] =  history_score[i]['total_rank'][j] 
       whe = np.where(np.mean(array_total_rank,axis=0) < 2)[0]
     
       print('GE < 2 : ',(np.min(whe) if whe.shape[0] >= 1 else self.traces_per_exp))        

       file = open(METRICS_FOLDER + 'history_attack_experiments_propagation_{}'.format(self.n_experiments),'wb')
       pickle.dump(history_score,file)
       file.close()


                
   
def run_attack():                
    attack = Attack()
    attack.run()
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--SINGLE_TASK', action="store_true", dest="SINGLE_TASK",
                        help='Single task models', default=False)
    parser.add_argument('--SINGLE_TASK_XOR',   action="store_true", dest="SINGLE_TASK_XOR", help='Single task xor mdoels', default=False)
    parser.add_argument('--MULTI',   action="store_true", dest="MULTI", help='Multi learning models', default=False)
    parser.add_argument('-scenario',   action="store", dest="TRAINING_TYPE", help='choose the input scenario', default='extracted')
    parser.add_argument('--ALL',   action="store_true", dest="ALL", help='All model types', default=False)
        
    args            = parser.parse_args()
  

    SINGLE_TASK        = args.SINGLE_TASK
    SINGLE_TASK_XOR        = args.SINGLE_TASK_XOR
    MULTI = args.MULTI
    ALL = args.ALL
    TRAINING_TYPE= args.TRAINING_TYPE
    print(TRAINING_TYPE)

    TARGETS = {}



    MODEL_TYPE = ['hierarchical']


    for model_type in MODEL_TYPE:
        for model_name in os.listdir(MODEL_FOLDER):
            if not model_name == 'model.h5':
                continue
    
            process_eval = Process(target=run_attack, args=()  )
            process_eval.start()
            process_eval.join()
                            
            
            
    
    
            
            
        
        