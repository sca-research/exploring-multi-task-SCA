from utility import read_from_h5_file  , get_hot_encode , load_model_from_name , get_rank 
from utility import XorLayer 
from utility import METRICS_FOLDER , MODEL_FOLDER
from train_models import model_single_task_xor    , model_multi_task, model_single_task_twin

import argparse , parse
from multiprocessing import Process
from tqdm import tqdm
import numpy as np
import pickle 
import os
        
        



class Attack:
    def __init__(self,convolution_blocks , kernel_size,filters,strides , pooling_size,dense_blocks,dense_units,n_experiments = 1000,n_traces = 10000,model_type = 'multi_task'):
        
        self.models = {}
        self.n_traces = n_traces

        id_model  = 'cb{}ks{}f{}s{}ps{}db{}du{}'.format(convolution_blocks , kernel_size,filters,strides , pooling_size,dense_blocks,dense_units)


        if 'multi_task' in model_type:
            multi_name = 'model_{}_all_{}.h5'.format(model_type,id_model) 
            model_struct = model_multi_task(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,input_length = 250000,summary = False)
            self.models['all'] = load_model_from_name(model_struct,multi_name)  
        else:
            for byte in range(6,7):
                name = 'model_{}_{}_{}.h5'.format(model_type,byte,id_model) 
                if 'xor' in name:
                    model_struct = model_single_task_xor(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,input_length = 250000,summary = False)                
                else:
                    model_struct = model_single_task_twin(convolution_blocks,dense_blocks , kernel_size,filters, strides , pooling_size,dense_units,input_length = 250000,summary = False)  
                self.models[byte] = load_model_from_name(model_struct,name) 

        self.n_experiments = n_experiments
        self.powervalues = {}

        traces , labels_dict, metadata  = read_from_h5_file(n_traces = self.n_traces,dataset = 'attack',load_plaintexts = True)
        traces = np.expand_dims(traces,2)
        
        self.correct_guesses = {}
        self.history_score = {}
        self.traces_per_exp = 1000
        self.predictions= np.zeros((self.n_traces,256),dtype =np.float32)
        plaintexts = np.array(metadata['plaintexts'],dtype = np.uint8)[:self.n_traces,6]
        self.plaintexts = get_hot_encode(plaintexts)
        batch_size = self.n_traces//10
  
        
        
        X = {}
        X['traces'] = traces
         
        predictions_p= np.zeros((self.n_traces,256),dtype =np.float32)
        if 'multi_task' in model_type:
            all_predictions = {}         
            all_predictions = self.models['all'].predict(X,verbose=1 ,batch_size = 250)    
            
            predictions_p = all_predictions['output_t_6']
        else:
            predictions_p = self.models[6].predict(X,verbose=1 ,batch_size = 250)['output']   
  
          
        for batch in tqdm(range(self.n_traces// batch_size)):
            self.predictions[batch_size*batch:batch_size*(batch +1)] = XorLayer()([predictions_p[batch_size*batch:batch_size*(batch +1)],self.plaintexts[batch_size*batch:batch_size*(batch +1)]])

        
        self.subkeys = 0x66
        
        
    def run(self,typ,id_model,print_logs = False):
       history_score = {}
       for experiment in tqdm(range(self.n_experiments)):
           if print_logs:
               print('====================')
               print('Experiment {} '.format(experiment))
           history_score[experiment] = {}
           history_score[experiment]['total_rank'] =  [] 
           subkeys_guess = {}
           subkeys_guess = np.zeros(256,)            
           
            
           traces_order = np.random.permutation(self.n_traces)[:self.traces_per_exp] 
           count_trace = 1
           
           for trace in traces_order:
               all_recovered = True
               ranks = {}
               if print_logs:
                   print('========= Trace {} ========='.format(count_trace))
               rank_string = ""

               subkeys_guess += np.log(self.predictions[trace] + 1e-36)
             
               
               ranks = get_rank(subkeys_guess,self.subkeys)
               rank_string += "| rank for byte {} : {} | \n".format('6',ranks)
               if np.argmax(subkeys_guess) == self.subkeys:
                    all_recovered = True                        
               else:                    
                    all_recovered = False                
              
               history_score[experiment]['total_rank'].append(ranks)
               if print_logs:
                   print(rank_string)
                   print('Total rank 2^{}'.format( history_score[experiment]['total_rank'][-1]))
                   print('\n')
               if all_recovered:  
                   if print_logs:
                       
                       print('All bytes Recovered at trace {}'.format(count_trace))
                   
                   for elem in range(count_trace,self.traces_per_exp):
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
       print(typ)
       print('GE < 2 : ',(np.min(whe) if whe.shape[0] >= 1 else self.traces_per_exp))        

       file = open(METRICS_FOLDER + 'history_attack_experiments_{}_{}_{}'.format(typ,id_model,self.n_experiments),'wb')
       pickle.dump(history_score,file)
       file.close()


   
def run_attack(convolution_blocks , kernel_size,filters,strides , pooling_size,dense_blocks,dense_units,model_type):                
    attack = Attack(convolution_blocks , kernel_size,filters,strides , pooling_size,dense_blocks,dense_units,model_type = model_type)
    id_model  = 'cb{}ks{}f{}s{}ps{}db{}du{}.h5'.format(convolution_blocks , kernel_size,filters,strides , pooling_size,dense_blocks,dense_units)
    attack.run('{}'.format(model_type),id_model)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--XOR',   action="store_true", dest="SINGLE_TASK_XOR", help='Adding the masks to the labels', default=False)
    parser.add_argument('--TWIN',   action="store_true", dest="SINGLE_TASK_TWIN", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI',   action="store_true", dest="MULTI", help='Adding the masks to the labels', default=False)
    #parser.add_argument('-scenario',   action="store", dest="TRAINING_TYPE", help='Adding the masks to the labels', default='extracted')
    parser.add_argument('--ALL',   action="store_true", dest="ALL", help='Adding the masks to the labels', default=False)
        
    args            = parser.parse_args()
  

   
    SINGLE_TASK_XOR        = args.SINGLE_TASK_XOR
    SINGLE_TASK_TWIN = args.SINGLE_TASK_TWIN
    MULTI = args.MULTI
    ALL = args.ALL


    TARGETS = {}

    if SINGLE_TASK_XOR:
        MODEL_TYPE = ['single_task_xor']
    elif SINGLE_TASK_TWIN:
        MODEL_TYPE = ['single_task_twin']
    elif MULTI:
        MODEL_TYPE = ['multi_task']
    elif ALL:
        MODEL_TYPE = ['single_task_xor','single_task_twin','multi_task']
    else:
        print('No training mode selected')

    for model_type in MODEL_TYPE:
        for model_name in os.listdir(MODEL_FOLDER):
            byte = 'all' if 'multi' in model_name else '6'
            multi_task =  'model_{}_{}'.format(model_type,byte)
      
            if not multi_task  in model_name :
                continue
            
            format_string = multi_task + '_cb{}ks{}f{}s{}ps{}db{}du{}.h5'
            parsed = parse.parse(format_string,model_name)
            convolution_blocks = int(parsed[0])
            kernel_size_list = parsed[1][1:-1]
            kernel_size_list = kernel_size_list.split(',')   
            kernel_size = [int(elem) for elem in kernel_size_list]
            filters = int(parsed[2])
            strides = int(parsed[3])
            pooling_size = int(parsed[4])
            dense_blocks = int(parsed[5])
            dense_units = int(parsed[6])
            process_eval = Process(target=run_attack, args=(convolution_blocks , kernel_size,filters,strides , pooling_size,dense_blocks,dense_units,model_type))
            process_eval.start()
            process_eval.join()
                            
            
            
    
    
            
            
        
        