from utility import read_from_h5_file , adapt_plaintexts, get_rank_list_from_prob_dist , get_hot_encode , load_model_from_name , get_rank , get_pow_rank
from utility import XorLayer , InvSboxLayer, MultiLayer
from utility import METRICS_FOLDER 
from gmpy2 import mpz,mul

from train_models import model_hierarchical, model_flat , model_alpha_single , model_rin_single , model_beta_single, model_intermediate_single

import argparse 
from multiprocessing import Process
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import pickle 


class Attack:
    def __init__(self,n_experiments = 1000,n_traces = 45000,model_type = 'multi_task'):
        

        self.n_experiments = n_experiments
        self.n_traces = n_traces
        self.powervalues = {}

        traces , labels_dict, metadata  = read_from_h5_file(n_traces = self.n_traces,dataset = 'validation',load_plaintexts = True)


        self.correct_guesses = {}
        self.history_score = {}
        self.traces_per_exp = 100
        self.predictions = np.zeros((16,self.n_traces,256),dtype =np.float32)
        
        
        plaintexts = np.array(metadata['plaintexts'],dtype = np.uint8)[:self.n_traces]
        keys =  np.array(metadata['keys'],dtype = np.uint8)[:self.n_traces]
        self.key = 0x00112233445566778899AABBCCDDEEFF
        master_key =[0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                      0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF ]  
        self.model_type = model_type  

        
        self.plaintexts = tf.cast(get_hot_encode(adapt_plaintexts(plaintexts,keys,self.key)),tf.float32)
        permutations = np.empty((self.n_traces,16,16),np.float32)
        for byte in range(16):
            permutations[:,byte] = get_hot_encode(np.array(labels_dict['p'],dtype = np.float32)[:self.n_traces,byte],classes = 16)
        
        batch_size = self.n_traces//10
  

        
        

 
        if model_type == 'hierarchical':
            multi_name = 'model_hierarchical_now_all2.h5'
            X_multi = {}
            model_struct , _ , _ , _ = model_hierarchical()
            self.model = load_model_from_name(model_struct,multi_name)  
            
            X_multi['inputs_alpha'] = traces[:,:2000]
            X_multi['inputs_rin'] = traces[:,2000:2000+1000]
            X_multi['inputs_beta'] = traces[:,3000:3000+200]

            X_multi['inputs_block'] = traces[:,4304:4304 + 93 * 16].reshape((n_traces,93,16))
            X_multi['inputs_permutations'] = traces[:,4304+ 93 * 16:4304+ 93 * 16 + 93 * 16].reshape((n_traces,93,16))
            predictions = self.model.predict(X_multi,batch_size  = 200)
            predictions_beta = predictions['beta']
            
            predictions_rin = predictions['rin']
            predictions_alpha = predictions['alpha']
            
            predictions_s_beta = np.empty((16,self.n_traces,256))
            predictions_t_rin = np.empty((16,self.n_traces,256))
            
            

          
            predictions_non_permuted = np.empty((16,self.n_traces,256),dtype = np.float32)
            for byte in range(16):
             
                predictions_non_permuted[byte] = predictions['kj_{}'.format(byte)]
                predictions_s_beta[byte] = predictions['s_beta_{}'.format(byte)]
                predictions_t_rin[byte] = predictions['t_rin_{}'.format(byte)]

            predictions_non_permuted = np.swapaxes(predictions_non_permuted,1,0)
            predictions_non_permuted = np.swapaxes(predictions_non_permuted,2,1)
           
            
            predictions = tf.matmul(predictions_non_permuted,permutations)
                         
            for batch in tqdm(range(self.n_traces// batch_size)):
                for byte in range(16):                   
            
                    self.predictions[byte][batch_size*batch:batch_size*(batch +1)] = XorLayer()([predictions[batch_size*batch:batch_size*(batch +1),:,byte],self.plaintexts[batch_size*batch:batch_size*(batch +1),byte]])
        elif model_type == 'flat':
            multi_name = 'model_flat_all2.h5'
            X_multi = {}
            model_struct , _ , _ , _ = model_flat()
            self.model = load_model_from_name(model_struct,multi_name)  
            
            X_multi['inputs_alpha'] = traces[:,:2000]
            X_multi['inputs_rin'] = traces[:,2000:2000+1000]
            X_multi['inputs_beta'] = traces[:,3000:3000+200]
            X_multi['inputs_block'] = traces[:,4304:4304 + 93 * 16].reshape((n_traces,93,16))
            predictions = self.model.predict(X_multi,batch_size  = 200)
            predictions_beta = predictions['beta']
            
            predictions_rin = predictions['rin']
            predictions_alpha = predictions['alpha']
            
            predictions_s_beta = np.empty((16,self.n_traces,256))
            predictions_t_rin = np.empty((16,self.n_traces,256))
            
            

          
            predictions_non_permuted = np.empty((16,self.n_traces,256),dtype = np.float32)
            for byte in range(16):
             
                
                predictions_s_beta[byte] = predictions['s_beta_{}'.format(byte)]
                predictions_t_rin[byte] = predictions['t_rin_{}'.format(byte)]

            predictions_non_permuted = np.empty((n_traces,256,16),dtype = np.float32)
            

            xor_layer = XorLayer(name = 'xor_layer')
            multi_layer = MultiLayer(name = 'multi_layer')
            inv_layer = InvSboxLayer(name = 'inv_layer')

            
            for batch in tqdm(range(self.n_traces// batch_size)):  
                for byte in range(16):
                    a_sj = xor_layer([predictions_s_beta[byte,batch_size*batch:batch_size*(batch +1)],predictions_beta[batch_size*batch:batch_size*(batch +1)]])
                    sj = multi_layer([a_sj,predictions_alpha[batch_size*batch:batch_size*(batch +1)]])
                    tj_from_sj = inv_layer(sj)
                    
                    a_tj = xor_layer([predictions_t_rin[byte,batch_size*batch:batch_size*(batch +1)],predictions_rin[batch_size*batch:batch_size*(batch +1)]])
                    tj = multi_layer([a_tj,predictions_alpha[batch_size*batch:batch_size*(batch +1)]])          
                    tj = (tj + tj_from_sj) / 2
                    predictions_non_permuted[batch_size*batch:batch_size*(batch +1),:,byte] = tj
                
            predictions = tf.matmul(predictions_non_permuted,permutations)
                         
            for batch in tqdm(range(self.n_traces// batch_size)):
                for byte in range(16):                   
            
                    self.predictions[byte][batch_size*batch:batch_size*(batch +1)] = XorLayer()([predictions[batch_size*batch:batch_size*(batch +1),:,byte],self.plaintexts[batch_size*batch:batch_size*(batch +1),byte]])
                                                              
    
        else:
            

            X_profiling_dict = {}
            X_profiling_dict['inputs_alpha'] = traces[:,:2000]
            X_profiling_dict['inputs_rin'] = traces[:,2000:2000+1000]
            X_profiling_dict['inputs_beta'] = traces[:,3000:3000+200]
            X_profiling_dict['inputs_block'] = traces[:,4304:4304 + 93 *16 ].reshape((-1,93,16))
            X_profiling_dict['inputs_permutations'] = traces[:,4304+93*16:4304 +2* 93 *16 ].reshape((-1,93,16))                               
            model_alpha , _ , _ , _  = model_alpha_single()    
            model_alpha = load_model_from_name(model_alpha, 'model_alpha2.h5')
            predictions_alpha = model_alpha.predict({'inputs_alpha':X_profiling_dict['inputs_alpha']})['output']
            model_rin , _ , _ , _  = model_rin_single()    
            model_rin = load_model_from_name(model_rin, 'model_rin2.h5')
            predictions_rin = model_rin.predict({'inputs_rin':X_profiling_dict['inputs_rin']})['output']
            model_beta , _ , _ , _  = model_beta_single() 
            model_beta = load_model_from_name(model_beta, 'model_beta2.h5')
            predictions_beta = model_beta.predict({'inputs_beta':X_profiling_dict['inputs_beta']})['output']
            predictions_t_rin =np.empty((16,self.n_traces,256),dtype = np.float32)
            predictions_s_beta = np.empty((16,self.n_traces,256),dtype = np.float32)
    
            for byte in range(16):
                byte_name = '0{}'.format('0' + str(byte+1) if byte+1 <10 else str(byte+1))
                model_s , _ , _ , _  = model_intermediate_single()  
                model_s = load_model_from_name(model_s, 'model_s1^beta_s{}^beta2.h5'.format(byte_name))
                predictions_s_beta[byte] = model_s.predict({'inputs_block':X_profiling_dict['inputs_block']})['output']
                
                model_t , _ , _ , _  = model_intermediate_single()
                model_t = load_model_from_name(model_t, 'model_t1^rin_t{}^rin2.h5'.format(byte_name))
                predictions_t_rin[byte] = model_t.predict({'inputs_block':X_profiling_dict['inputs_block']})['output']
                

            predictions_non_permuted = np.empty((n_traces,256,16),dtype = np.float32)
            

            xor_layer = XorLayer(name = 'xor_layer')
            multi_layer = MultiLayer(name = 'multi_layer')
            inv_layer = InvSboxLayer(name = 'inv_layer')

            
            for batch in tqdm(range(self.n_traces// batch_size)):  
                for byte in range(16):
                    a_sj = xor_layer([predictions_s_beta[byte,batch_size*batch:batch_size*(batch +1)],predictions_beta[batch_size*batch:batch_size*(batch +1)]])
                    sj = multi_layer([a_sj,predictions_alpha[batch_size*batch:batch_size*(batch +1)]])
                    tj_from_sj = inv_layer(sj)
                    
                    a_tj = xor_layer([predictions_t_rin[byte,batch_size*batch:batch_size*(batch +1)],predictions_rin[batch_size*batch:batch_size*(batch +1)]])
                    tj = multi_layer([a_tj,predictions_alpha[batch_size*batch:batch_size*(batch +1)]])          
                    tj = (tj + tj_from_sj) / 2
                    predictions_non_permuted[batch_size*batch:batch_size*(batch +1),:,byte] = tj
                
            predictions = tf.matmul(predictions_non_permuted,permutations)
                         
            for batch in tqdm(range(self.n_traces// batch_size)):
                for byte in range(16):                   
            
                    self.predictions[byte][batch_size*batch:batch_size*(batch +1)] = XorLayer()([predictions[batch_size*batch:batch_size*(batch +1),:,byte],self.plaintexts[batch_size*batch:batch_size*(batch +1),byte]])
                    
          
        master_key = np.array(master_key,dtype = np.int32)
        self.subkeys = master_key
        _ , acc , _, _  = get_rank_list_from_prob_dist(predictions_alpha,labels_dict['alpha'])        
        print('Accuracy for {}'.format("alpha"), acc) 
        _ , acc , _, _  = get_rank_list_from_prob_dist(predictions_rin,labels_dict['rin'])
        
        print('Accuracy for {}'.format("rin"), acc) 
        _ , acc , _, _  = get_rank_list_from_prob_dist(predictions_beta,labels_dict['beta'])
        
        print('Accuracy for {}'.format("beta"), acc) 
        
        
        
        acc_k = 0
        acc_t_rin = 0
        acc_s_beta =0 

        for byte in range(16):
            rank , acc , score , _  = get_rank_list_from_prob_dist(self.predictions[byte],np.repeat(self.subkeys[byte],self.predictions.shape[1]))
            acc_k += acc
                  
            _ , acc , _, _  = get_rank_list_from_prob_dist(predictions_s_beta[byte],np.array(labels_dict['s1^beta'],dtype = np.uint8)[:,byte])
            acc_s_beta+=acc 
            _ , acc , _, _  = get_rank_list_from_prob_dist(predictions_t_rin[byte],np.array(labels_dict['t1^rin'],dtype = np.uint8)[:,byte])
            acc_t_rin+=acc 

        print('Average accuracy k ',acc_k/16)
        print('Average accuracy s^beta ',acc_s_beta/16)
        print('Average accuracy t_rin ',acc_t_rin/16)
    
        


        
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
       print(np.mean(array_total_rank,axis=0))
     
       print('GE < 2 : ',(np.min(whe) if whe.shape[0] >= 1 else self.traces_per_exp))        

       file = open(METRICS_FOLDER + 'history_attack_experiments_{}_{}'.format(self.model_type,self.n_experiments),'wb')
       pickle.dump(history_score,file)
       file.close()


                
   
def run_attack(model_type):                
    attack = Attack(model_type= model_type)
    attack.run()
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--FLAT', action="store_true", dest="FLAT",
                        help='Single task models', default=False)
    parser.add_argument('--HIERARCHICAL', action="store_true", dest="HIERARCHICAL",
                        help='Single task models', default=False)
    args            = parser.parse_args()
  

    FLAT        = args.FLAT
    HIERARCHICAL = args.HIERARCHICAL


    TARGETS = {}



    MODEL_TYPE = ['classical']
    if FLAT:
        MODEL_TYPE = ['flat']
    if HIERARCHICAL:
        MODEL_TYPE = ['hierarchical']

    for model_type in MODEL_TYPE:

        process_eval = Process(target=run_attack, args=(model_type,)  )
        process_eval.start()
        process_eval.join()
                            
            
            
    
    
            
            
        
        