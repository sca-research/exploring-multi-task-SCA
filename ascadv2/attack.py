from utility import read_from_h5_file, get_rank_list_from_prob_dist , adapt_plaintexts , get_hot_encode , load_model_from_name , get_rank , get_pow_rank
from utility import XorLayer , InvSboxLayer
from utility import METRICS_FOLDER 
from gmpy2 import mpz,mul

from train_models import model_multi, model_single_task, model_multi_task_t_only, model_multi_task_s_only

import argparse 
from multiprocessing import Process
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import pickle 


class Attack:
    def __init__(self,n_experiments = 1000,n_traces = 5000,model_type = 'multi_task', multi_model = False, shared = False):
        self.name ='{}_{}_{}'.format('multi_model' if multi_model else 'model',model_type, 'shared' if shared else 'nshared')
        self.models = {}
        print(model_type)
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
        if 'multi' == model_type:
            model_t = '{}_{}_{}2.h5'.format('multi_model' if multi_model else 'model',model_type, 'shared' if shared else 'nshared')
            model_struct , _ , _  = model_multi(summary = False,shared = shared,multi_model = multi_model)
            self.models['all'] = load_model_from_name(model_struct,model_t)  
            alpha_known = False
        elif 'multi_t' in model_type:
            model_t = '{}_{}_{}2.h5'.format('multi_model' if multi_model else 'model',model_type, 'shared' if shared else 'nshared')
            known_alpha = 'first' in model_type
            model_struct , _ , _  = model_multi_task_t_only(shared = shared,multi_model = multi_model,known_alpha = known_alpha,summary = False)
            self.models['all'] = load_model_from_name(model_struct,model_t)  
            alpha_known = True
        elif 'multi_s' in model_type:
            known_alpha = 'first' in model_type
            model_t = '{}_{}_{}2.h5'.format('multi_model' if multi_model else 'model',model_type, 'shared' if shared else 'nshared')
            model_struct , _ , _  = model_multi_task_s_only(shared = shared,multi_model = multi_model,known_alpha = known_alpha,summary = False)
            self.models['all'] = load_model_from_name(model_struct,model_t)  
            alpha_known = True
                                   
        else:   
            alpha_known = 'first' in model_type
            for byte in range(16):
                
                model_t = 'model_{}_nshared{}2.h5'.format(model_type,'_'+str(byte) if not (byte is None) else '')
                model_struct , _ , _  = model_single_task(s =  'single_s' in model_type,t = 'single_t' in model_type, alpha_known = alpha_known, summary = False)
                self.models[byte] = load_model_from_name(model_struct,model_t)  

        X_multi = {}

        X_multi['inputs_alpha'] = traces[:,:2000]
        X_multi['alpha'] = get_hot_encode(np.array(labels_dict['alpha'][:n_traces],dtype = np.uint8))
        X_multi['inputs_rin'] = traces[:,2000:2000+1000]
        X_multi['inputs_beta'] = traces[:,3000:3000+200]
        X_multi['inputs_block'] = traces[:,4304:4304 + 93 * 16].reshape((n_traces,93,16))
        non_permuted_predictions= np.empty((self.n_traces,16,256),dtype = np.float32)
        inv_sbox = InvSboxLayer(name = 'inv_sbox')
        if 'multi_t' in model_type:
            predictions = self.models['all'].predict(X_multi,batch_size  = 200)
            for byte in range(16):  
                tj = predictions['output_{}'.format(byte)]             
                non_permuted_predictions[:,byte] = tj
        elif 'multi_s' in model_type:
            predictions = self.models['all'].predict(X_multi,batch_size  = 200)
            for byte in range(16):
                sj = predictions['output_{}'.format(byte)]
                tj = inv_sbox(sj)
                non_permuted_predictions[:,byte] = tj
        elif 'multi' in model_type:
            predictions = self.models['all'].predict(X_multi,batch_size  = 200)
            for byte in range(16):
                sj = predictions['output_sj_{}'.format(byte)]
                tj = predictions['output_tj_{}'.format(byte)]
                tj *= inv_sbox(sj)
                non_permuted_predictions[:,byte] = tj        
        else:
            for byte in range(16):
                predictions = self.models[byte].predict(X_multi,batch_size = 200)
                x = predictions['output']
                if 'single_s' in model_type:
                    tj = inv_sbox(x)
                else:
                    tj = x
                non_permuted_predictions[:,byte] = tj            
        permuted_predictions= np.empty((self.n_traces,16,256),dtype = np.float32)
        for elem in range(self.n_traces):
            for byte in range(16):
                permuted_predictions[elem,self.permutations[elem,byte]] = non_permuted_predictions[elem,byte]
           
        for batch in tqdm(range(self.n_traces//batch_size)):
            for byte in range(16):                   
                
                self.predictions[byte][batch_size*batch:batch_size*(batch +1)] = XorLayer()([permuted_predictions[batch_size*batch:batch_size*(batch +1),byte] ,self.plaintexts[batch_size*batch:batch_size*(batch +1),byte]])

       
        master_key = np.array(master_key,dtype = np.int32)
        self.subkeys = master_key
        
        
        
    def evaluate(self):
        accuracies = []
        for byte in range(16):
            _ , acc , _, _  = get_rank_list_from_prob_dist(self.predictions[byte],np.repeat(self.subkeys[byte],self.predictions.shape[1]))
            accuracies.append(acc)
        print(min(accuracies))
        print(max(accuracies))
        print(np.mean(accuracies))
        
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

       array_total_rank = np.empty((self.n_experiments,self.traces_per_exp))
       for i in range(self.n_experiments):
           for j in range(self.traces_per_exp):
               array_total_rank[i][j] =  history_score[i]['total_rank'][j] 
       whe = np.where(np.mean(array_total_rank,axis=0) < 2)[0]
     
       print('GE < 2 : ',(np.min(whe) if whe.shape[0] >= 1 else self.traces_per_exp))        
       print('GE 1T : ',np.mean(array_total_rank,axis = 0)[0])
       file = open(METRICS_FOLDER + 'history_attack_experiments_{}_{}'.format(self.name,self.n_experiments),'wb')
       pickle.dump(history_score,file)
       file.close()
       

                
   
def run_attack(model_type, multi_model,shared):                
    attack = Attack(model_type=model_type, shared = shared, multi_model = multi_model)
    attack.run()
    attack.evaluate()

    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--SHARED', action="store_true", dest="SHARED",
                        help='Hard parameter sharing', default=False)
    parser.add_argument('--MULTI_MODEL',   action="store_true", dest="MULTI_MODEL", help='L_reg loss function', default=False)


        
    args            = parser.parse_args()
  

    SHARED        = args.SHARED
    MULTI_MODEL        = args.MULTI_MODEL


    TARGETS = {}

    

    MODEL_TYPE = [ 'multi_t','multi_s','multi_t_first','multi_s_first','multi','single_t_first','single_s_first']


    for model_type in MODEL_TYPE:
        process_eval = Process(target=run_attack, args=(model_type,MULTI_MODEL,SHARED)  )
        process_eval.start()
        process_eval.join()
                            
            
            
    
    
            
            
        
        