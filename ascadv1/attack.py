from utility import read_from_h5_file  , get_hot_encode , load_model_from_name , get_rank , get_pow_rank, get_rank_list_from_prob_dist
from utility import XorLayer , InvSboxLayer
from utility import METRICS_FOLDER 
from train_models import model_multi_task_single_target,model_single_task    , model_multi_task_single_target_one_shared_mask, model_multi_task_single_target_not_shared,model_multi_task_single_target_one_shared_mask_shared_branch
from gmpy2 import mpz,mul
import argparse 
from multiprocessing import Process
from tqdm import tqdm
import numpy as np
import pickle 
   



class Attack:
    def __init__(self,training_type, n_experiments = 1000,n_traces = 10000,model_type = False):
        
        self.models = {}
        self.n_traces = n_traces

        if training_type == 'single_task_subin' or training_type == 'single_task_subout':
            for byte in range(2,16):
                model = model_single_task(input_length = 250000,summary = False)   
                model = load_model_from_name(model,'model_{}_{}.h5'.format(training_type,byte))
                self.models[byte] = model
            
            target = 's' if 'subout' in training_type else 't'
            self.name = training_type


        elif training_type == 'multi_task_single_target':
    
            model = model_multi_task_single_target(multi_model = model_type,input_length = 250000)                 
            target = 's'
  
        elif training_type == 'multi_task_single_target_not_shared':
    
            model = model_multi_task_single_target_not_shared(multi_model = model_type,input_length = 250000)  
            target = 's'

        elif training_type == 'multi_task_single_target_one_shared_mask':
            
            model = model_multi_task_single_target_one_shared_mask(multi_model = model_type,input_length = 250000)   
            target = 't'               

        elif training_type == 'multi_task_single_target_one_shared_mask_shared_branch':
            
            model = model_multi_task_single_target_one_shared_mask_shared_branch(multi_model = model_type,input_length = 250000)    
            target = 't'                   
    
        else:
            print('Some error here')       


        self.n_experiments = n_experiments
        self.powervalues = {}

        traces , labels_dict, metadata  = read_from_h5_file(n_traces = self.n_traces,dataset = 'attack',load_plaintexts = True)
        
        
        self.correct_guesses = {}
        self.history_score = {}
        self.traces_per_exp = 100
        
        plaintexts = np.array(metadata['plaintexts'],dtype = np.uint8)[:self.n_traces]
        self.plaintexts = get_hot_encode(plaintexts)
        batch_size = self.n_traces//10
  
        
        
     
        master_key = [0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xAA,0xBB,0xCC,0xDD,0xEE,0xFF]
        self.subkeys = master_key

        self.powervalues = np.expand_dims(traces,2)
        predictions = {}
        if 'single_task' in training_type:
            for byte in range(2,16):
                predictions['output_{}'.format(byte)] = self.models[byte].predict({'traces':self.powervalues})['output']
        else:
            self.name = '{}_{}'.format('model' if not model_type else 'multi_model',training_type)
            model = load_model_from_name(model, self.name + '_all.h5')

            predictions = model.predict({'traces':self.powervalues})
        self.predictions = np.empty((self.n_traces, 14,256),dtype =np.float32) 
        batch_size = 1000
        
        xor_op = XorLayer(name = 'xor')
        inv_op = InvSboxLayer(name = 'iinv')
        for batch in range(0,self.n_traces//batch_size):
            print('Batch of prediction {} / {}'.format(batch + 1,self.n_traces//batch_size))
            for byte in tqdm(range(2,16)):                  
                    if target == 't': 
                        self.predictions[batch*batch_size:(batch+1)*batch_size,byte-2] = xor_op([predictions['output_{}'.format(byte)][batch*batch_size:(batch+1)*batch_size],self.plaintexts[batch*batch_size:(batch+1)*batch_size,byte]])
                    else:
                        inv_pred = inv_op(predictions['output_{}'.format(byte)][batch*batch_size:(batch+1)*batch_size])
                        self.predictions[batch*batch_size:(batch+1)*batch_size,byte-2] = xor_op([inv_pred,self.plaintexts[batch*batch_size:(batch+1)*batch_size,byte]])
                    
        for byte in range(14):
            _ , acc , _, _  = get_rank_list_from_prob_dist(self.predictions[:,byte],np.repeat(self.subkeys[byte+2],self.predictions.shape[0]))
            print('Accuracy for byte {}'.format(byte + 2 ), acc)   
        
    def run(self,print_logs = False):
    
       for experiment in range(self.n_experiments):
           print('====================')
           print('Experiment {} '.format(experiment))
           self.history_score[experiment] = {}
           self.history_score[experiment]['total_rank'] =  [] 
           self.subkeys_guess = {}
           for i in range(2,16):
               self.subkeys_guess[i] = np.zeros(256,)            
           
               self.history_score[experiment][i] = []
           traces_order = np.random.permutation(self.n_traces)[:self.traces_per_exp] 
           count_trace = 1
           
           for trace in traces_order:
               
               
               
               recovered  = {}
               all_recovered = True
               ranks = {}

               print('========= Trace {} ========='.format(count_trace))
               rank_string = ""
               total_rank = mpz(1)
               
               
               for byte in range(2,16):
                   self.subkeys_guess[byte] += np.log(self.predictions[trace][byte-2] + 1e-36)
                  
                   ranks[byte-2] = get_rank(self.subkeys_guess[byte],self.subkeys[byte])
                   self.history_score[experiment][byte].append(ranks[byte-2])
                   total_rank = mul(total_rank,mpz(ranks[byte-2]))
                   rank_string += "| rank for byte {} : {} | \n".format(byte,ranks[byte-2])
                   if np.argmax(self.subkeys_guess[byte]) == self.subkeys[byte]:
                       recovered[byte] = True                        
                   else:
                       recovered[byte] = False
                       all_recovered = False                
              
               self.history_score[experiment]['total_rank'].append(get_pow_rank(total_rank))
               print(rank_string)
               print('Total rank 2^{}'.format( self.history_score[experiment]['total_rank'][-1]))
               print('\n')
               if all_recovered:                    
                   print('All bytes Recovered at trace {}'.format(count_trace))
                   
                   for elem in range(count_trace,self.traces_per_exp):
                       for i in range(2,16):
                           self.history_score[experiment][i].append(ranks[i-2])
                       self.history_score[experiment]['total_rank'].append(0)
                   break
                   count_trace += 1
               else:
                   count_trace += 1
               print('\n')
       array_total_rank = np.empty((self.n_experiments,self.traces_per_exp))
       for i in range(self.n_experiments):
           for j in range(self.traces_per_exp):
               array_total_rank[i][j] =  self.history_score[i]['total_rank'][j] 
       whe = np.where(np.mean(array_total_rank,axis=0) < 2)[0]
       print(np.mean(array_total_rank,axis=0))
       print('GE < 2 : ',(np.min(whe) if whe.shape[0] >= 1 else self.traces_per_exp))        

       file = open(METRICS_FOLDER + 'history_attack_experiments_{}_{}'.format(self.name,self.n_experiments),'wb')
       pickle.dump(self.history_score,file)
       file.close()


   
def run_attack(training_type,model_type):                
    attack = Attack(training_type,model_type = model_type)    
    attack.run()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--MULTI_MODEL', action="store_true", dest="MULTI_MODEL",
                        help='Classical training of the intermediates', default=False)
    parser.add_argument('--SINGLE',   action="store_true", dest="SINGLE", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI',   action="store_true", dest="MULTI", help='Adding the masks to the labels', default=False)
    args            = parser.parse_args()
  

    MULTI_MODEL        = args.MULTI_MODEL
    SINGLE        = args.SINGLE
    MULTI = args.MULTI

 

    TARGETS = {}
    if SINGLE:
        training_types = ['multi_task_single_target']

    elif MULTI:
        training_types = ['single_task_subin','single_task_subout']
    else:
        print('No training mode selected')


    for training_type in training_types:
        process_eval = Process(target=run_attack, args=(training_type,MULTI_MODEL))
        process_eval.start()
        process_eval.join()   
            
            
    
    
            
            
        
        