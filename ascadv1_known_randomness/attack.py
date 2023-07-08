# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 11:32:19 2022

@author: martho
"""
from utility import load_model_from_name , VARIABLE_LIST , XorLayer, read_from_h5_file , KEY_FIXED, get_hot_encode , get_rank , get_pow_rank, METRICS_FOLDER  , get_rank_list_from_prob_dist
import pickle
from gmpy2 import mpz,mul

from train_models import cnn_best,cnn_hierarchical_multi_target,cnn_multi_task_multi_target,cnn_multi_task_subbytes_inputs,cnn_hierarchical_subbytes_inputs
import argparse
from tqdm import tqdm
import numpy as np


class Attack:
    def __init__(self,n_experiments = 1,training_type = 'classical',target = 'k'):
        
        self.models = {}
        self.target = target
        traces , labels_dict, plaintexts = read_from_h5_file(dataset = 'attack',load_plaintexts = True)
        
        if training_type == 'classical':
            if self.target == 't':  
                model_struct =  cnn_best(input_length=250000)
                self.models['i'] = load_model_from_name(model_struct,'i_cnn_single.h5') 
                self.models['t1_i'] = {}
                self.models['r'] = {}
                self.models['t1_ri'] = {}
                self.models['t1_r'] = {}
            else:
                model_struct =  cnn_best(input_length=250000)
                self.models['i'] = load_model_from_name(model_struct,'i_cnn_single.h5') 
                self.models['s1_r'] = {}
                self.models['r'] = {}
                self.models['t1_i'] = {}
            for byte in range(2,16):
                for name  in self.models.keys():
                    if name == 'i':
                        continue
                    model_struct =  cnn_best(input_length=250000)
                    self.models[name][byte] = load_model_from_name(model_struct,'{}_cnn_single.h5'.format(VARIABLE_LIST[name.replace('_','^')][byte])) 
                    
        elif training_type == 'multi':
            
            self.models['propagation'] = {}
            for byte in range(2,16):
                if self.target == 'k':  
                    model_struct_propagation = cnn_multi_task_multi_target(input_length=250000)
                    self.models['propagation'][byte] = load_model_from_name(model_struct_propagation,'{}_cnn_multi_task_multi_target.h5'.format(VARIABLE_LIST['k1'][byte]))
                
                else:
                    model_struct_propagation = cnn_multi_task_subbytes_inputs(input_length=250000)
                    self.models['propagation'][byte] = load_model_from_name(model_struct_propagation,'{}_cnn_multi_task_subbytes_inputs.h5'.format(VARIABLE_LIST['t1'][byte]))
                
        else:
            
            self.models['propagation'] = {}
            for byte in range(2,16):
                if self.target == 'k':  
                    model_struct_propagation = cnn_hierarchical_multi_target(input_length=250000)
                    self.models['propagation'][byte] = load_model_from_name(model_struct_propagation,'{}_cnn_hierarchical_multi_target.h5'.format(VARIABLE_LIST['k1'][byte]))
                
                else:
                    model_struct_propagation = cnn_hierarchical_subbytes_inputs(input_length=250000)
                    self.models['propagation'][byte] = load_model_from_name(model_struct_propagation,'{}_cnn_hierarchical_subbytes_inputs.h5'.format(VARIABLE_LIST['t1'][byte]))
                
        self.n_experiments = n_experiments
        self.powervalues = {}
        
        mapping = (
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
        
        
        self.correct_guesses = {}
        self.history_score = {}
        self.traces_per_exp = 100
        self.n_total_attack_traces = 10000
        self.predictions = np.zeros((14,self.n_total_attack_traces,256))

        if not (training_type == 'hierarchical'):
                       
            predictions_r = np.empty((14,self.n_total_attack_traces,256))
            
            if self.target == 's': 
                predictions_s1_r= np.empty((14,self.n_total_attack_traces,256))
            if self.target == 't': 
                predictions_t1_i = np.empty((14,self.n_total_attack_traces,256)) 
                predictions_i = np.empty((self.n_total_attack_traces,256))
                predictions_t1_r= np.empty((14,self.n_total_attack_traces,256))
                predictions_t1_ri= np.empty((14,self.n_total_attack_traces,256))                
            else:
                predictions_t1_i = np.empty((14,self.n_total_attack_traces,256)) 
                predictions_i = np.empty((self.n_total_attack_traces,256))
                predictions_s1_r= np.empty((14,self.n_total_attack_traces,256))

            
        self.plaintexts = np.array(labels_dict['p1'],dtype = np.uint8)[:self.n_total_attack_traces]
        
        self.key = KEY_FIXED
        master_key = [0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xAA,0xBB,0xCC,0xDD,0xEE,0xFF]
        

        self.powervalues = np.expand_dims(traces[:self.n_total_attack_traces],2)
        batch_size = 1000
        for batch in range(0,self.n_total_attack_traces//batch_size):
            print('Batch of prediction {} / {}'.format(batch + 1,self.n_total_attack_traces//batch_size))
            for byte in tqdm(range(2,16)):                  
                
                
                if training_type == 'hierarchical':
                    if self.target == 'k': 
                        self.predictions[byte-2,batch*batch_size:(batch+1)*batch_size] = self.models['propagation'][byte].predict({'traces':self.powervalues[batch*batch_size:(batch+1)*batch_size], 'plaintext' : get_hot_encode(self.plaintexts[batch*batch_size:(batch+1)*batch_size,byte]) },verbose=0 ,batch_size = 250)['output']
                    else:
                        self.predictions[byte-2,batch*batch_size:(batch+1)*batch_size] = XorLayer()([ self.models['propagation'][byte].predict({'traces':self.powervalues[batch*batch_size:(batch+1)*batch_size] },verbose=0 ,batch_size = 250)['output'],get_hot_encode(self.plaintexts[batch*batch_size:(batch+1)*batch_size,byte])])     
                elif training_type == 'multi':
                    all_predictions = self.models['propagation'][byte] .predict({'traces':self.powervalues[batch*batch_size:(batch+1)*batch_size] },verbose=0 ,batch_size = 250)                  
                    predictions_r[byte-2,batch*batch_size:(batch+1)*batch_size] = all_predictions['output_r']
                    
                    if self.target == 's': 
                        predictions_s1_r[byte-2,batch*batch_size:(batch+1)*batch_size] = all_predictions['output_s1_r']
                        predictions_s1_from_r = XorLayer()([predictions_s1_r[byte-2,batch*batch_size:(batch+1)*batch_size],predictions_r[byte-2,batch*batch_size:(batch+1)*batch_size]])
                        predictions_sum_t1 = predictions_s1_from_r[:,mapping]
                    elif self.target == 't': 
                        if byte == 2:
                            predictions_i[batch*batch_size:(batch+1)*batch_size] = all_predictions['output_i']
                        predictions_t1_i[byte-2,batch*batch_size:(batch+1)*batch_size] = all_predictions['output_t1_i']  
                        predictions_t1_r[byte-2,batch*batch_size:(batch+1)*batch_size] = all_predictions['output_t1_r']
                        predictions_t1_ri[byte-2,batch*batch_size:(batch+1)*batch_size] = all_predictions['output_t1_ri']
                        predictions_ri = XorLayer()([predictions_r[byte-2,batch*batch_size:(batch+1)*batch_size],predictions_i[batch*batch_size:(batch+1)*batch_size]])
                        predictions_t1_from_i = XorLayer()([predictions_t1_i[byte-2,batch*batch_size:(batch+1)*batch_size],predictions_i[batch*batch_size:(batch+1)*batch_size]])
                        predictions_t1_from_r = XorLayer()([predictions_t1_r[byte-2,batch*batch_size:(batch+1)*batch_size],predictions_r[byte-2,batch*batch_size:(batch+1)*batch_size]])
                        predictions_t1_from_ri = XorLayer()([predictions_t1_ri[byte-2,batch*batch_size:(batch+1)*batch_size],predictions_ri])
                        predictions_sum_t1 = predictions_t1_from_i + predictions_t1_from_r + predictions_t1_from_ri

                    else:
                        if byte == 2:
                            predictions_i[batch*batch_size:(batch+1)*batch_size] = all_predictions['output_i']
                        predictions_t1_i[byte-2,batch*batch_size:(batch+1)*batch_size] = all_predictions['output_t1_i']  
                        predictions_s1_r[byte-2,batch*batch_size:(batch+1)*batch_size] = all_predictions['output_s1_r']
                        predictions_s1 = XorLayer()([predictions_s1_r[byte-2,batch*batch_size:(batch+1)*batch_size],predictions_r[byte-2,batch*batch_size:(batch+1)*batch_size]])
                        predictions_t1_from_s1 = predictions_s1[:,mapping]
                        predictions_t1 = XorLayer()([predictions_t1_i[byte-2,batch*batch_size:(batch+1)*batch_size],predictions_i[batch*batch_size:(batch+1)*batch_size]])
                        predictions_sum_t1 = predictions_t1_from_s1 +  predictions_t1                    
                    
                    self.predictions[byte-2,batch*batch_size:(batch+1)*batch_size] = XorLayer()([predictions_sum_t1,get_hot_encode(self.plaintexts[batch*batch_size:(batch+1)*batch_size,byte])])              
                else:

                    predictions_r[byte-2,batch*batch_size:(batch+1)*batch_size] = self.models['r'][byte] .predict({'traces':self.powervalues[batch*batch_size:(batch+1)*batch_size]},verbose=0)['output']
                   
                    if self.target == 's': 
                        predictions_s1_r[byte-2,batch*batch_size:(batch+1)*batch_size] = self.models['s1_r'][byte] .predict({'traces':self.powervalues[batch*batch_size:(batch+1)*batch_size]},verbose=0)['output']
                        predictions_s1 = XorLayer()([predictions_s1_r[byte-2,batch*batch_size:(batch+1)*batch_size],predictions_r[byte-2,batch*batch_size:(batch+1)*batch_size]])
                        predictions_t1_from_s1 = predictions_s1[:,mapping]
                        predictions_sum_t1 = predictions_t1_from_s1 
                    if self.target == 't': 
                        if byte == 2:
                            predictions_i[batch*batch_size:(batch+1)*batch_size] = self.models['i'].predict({'traces':self.powervalues[batch*batch_size:(batch+1)*batch_size]},verbose=0)['output']
                            
                        predictions_t1_i[byte-2,batch*batch_size:(batch+1)*batch_size] = self.models['t1_i'][byte] .predict({'traces':self.powervalues[batch*batch_size:(batch+1)*batch_size]},verbose=0)['output']
                        
                        predictions_t1_r[byte-2,batch*batch_size:(batch+1)*batch_size] = self.models['t1_r'][byte] .predict({'traces':self.powervalues[batch*batch_size:(batch+1)*batch_size]},verbose=0)['output']
                        predictions_t1_ri[byte-2,batch*batch_size:(batch+1)*batch_size] = self.models['t1_ri'][byte] .predict({'traces':self.powervalues[batch*batch_size:(batch+1)*batch_size]},verbose=0)['output']
                        predictions_ri = XorLayer()([predictions_r[byte-2,batch*batch_size:(batch+1)*batch_size],predictions_i[batch*batch_size:(batch+1)*batch_size]])
                        predictions_t1_from_i = XorLayer()([predictions_t1_i[byte-2,batch*batch_size:(batch+1)*batch_size],predictions_i[batch*batch_size:(batch+1)*batch_size]])
                        predictions_t1_from_r = XorLayer()([predictions_t1_r[byte-2,batch*batch_size:(batch+1)*batch_size],predictions_r[byte-2,batch*batch_size:(batch+1)*batch_size]])
                        predictions_t1_from_ri = XorLayer()([predictions_t1_ri[byte-2,batch*batch_size:(batch+1)*batch_size],predictions_ri])
                        predictions_sum_t1 = predictions_t1_from_i + predictions_t1_from_r + predictions_t1_from_ri

                    else:
                        if byte == 2:
                            predictions_i[batch*batch_size:(batch+1)*batch_size] = self.models['i'].predict({'traces':self.powervalues[batch*batch_size:(batch+1)*batch_size]},verbose=0)['output']
                            
                        predictions_t1_i[byte-2,batch*batch_size:(batch+1)*batch_size] = self.models['t1_i'][byte] .predict({'traces':self.powervalues[batch*batch_size:(batch+1)*batch_size]},verbose=0)['output']
                        
                        predictions_s1_r[byte-2,batch*batch_size:(batch+1)*batch_size] = self.models['s1_r'][byte] .predict({'traces':self.powervalues[batch*batch_size:(batch+1)*batch_size]},verbose=0)['output']
                        predictions_s1 = XorLayer()([predictions_s1_r[byte-2,batch*batch_size:(batch+1)*batch_size],predictions_r[byte-2,batch*batch_size:(batch+1)*batch_size]])
                        predictions_t1_from_s1 = predictions_s1[:,mapping]
                        predictions_t1 = XorLayer()([predictions_t1_i[byte-2,batch*batch_size:(batch+1)*batch_size],predictions_i[batch*batch_size:(batch+1)*batch_size]])
                        predictions_sum_t1 = predictions_t1_from_s1 +  predictions_t1                    
                    
                    self.predictions[byte-2,batch*batch_size:(batch+1)*batch_size] = XorLayer()([predictions_sum_t1,get_hot_encode(self.plaintexts[batch*batch_size:(batch+1)*batch_size,byte])])       
                

        
        self.subkeys = master_key[2:]

        


        
    def run(self):
       
       for experiment in range(self.n_experiments):
           print('====================')
           print('Experiment {} '.format(experiment))
           self.history_score[experiment] = {}
           self.history_score[experiment]['total_rank'] =  [] 
           self.subkeys_guess = {}
           for i in range(2,16):
               self.subkeys_guess[i] = np.zeros(256,)            
           
               self.history_score[experiment][i] = []
           traces_order = np.random.permutation(self.n_total_attack_traces)[:self.traces_per_exp] 
           count_trace = 1
           
           for trace in traces_order:
               
               
               
               recovered  = {}
               all_recovered = True
               ranks = {}

               print('========= Trace {} ========='.format(count_trace))
               rank_string = ""
               total_rank = mpz(1)
               ranking = np.empty((14,256),dtype = np.float32)
               
               for byte in range(2,16):
                   self.subkeys_guess[byte] += np.log(self.predictions[byte-2][trace] + 1e-36)
                  
                   ranks[byte-2] = get_rank(self.subkeys_guess[byte],self.subkeys[byte-2])
                   self.history_score[experiment][byte].append(ranks[byte-2])
                   total_rank = mul(total_rank,mpz(ranks[byte-2]))
                   rank_string += "| rank for byte {} : {} | \n".format(byte,ranks[byte-2])
                   if np.argmax(self.subkeys_guess[byte]) == self.subkeys[byte-2]:
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
               

       file = open(METRICS_FOLDER + 'history_attack_on_{}_experiments_{}_{}'.format( self.target ,training_type,self.n_experiments),'wb')
       pickle.dump(self.history_score,file)
       file.close()
       
       for byte in range(14):
            _ , acc , _, _  = get_rank_list_from_prob_dist(self.predictions[byte],np.repeat(self.subkeys[byte],self.predictions.shape[1]))
            print('Accuracy for byte {}'.format(byte + 2 ), acc)
        
            
                
                
                    
        # file = open('history_attack_experiments_{}'.format(self.n_experiments),'wb')
        # pickle.dump(self.history_score,file)
        # file.close()
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')

    parser.add_argument('-e', '-experiment', action="store", dest="EXPERIMENT", help='Number of Epochs in Training (default: 75 CNN, 100 MLP)',
                        type=int, default=1000)
    parser.add_argument('--INDIV', action="store_true", dest="INDIV", help='for attack dataset', default=False)
    parser.add_argument('--MULTI', action="store_true", dest="MULTI", help='for attack dataset', default=False)
    parser.add_argument('-t', action="store", dest="T", help='for attack dataset',type = str, default='k')
    args            = parser.parse_args()
    
    


    EXPERIMENT = args.EXPERIMENT
    INDIV = args.INDIV
    MULTI = args.MULTI
    T = args.T
    training_type = 'hierarchical'
    if INDIV : 
        training_type = 'classical'
    if MULTI :
        training_type = 'multi'
    print(training_type)
    attack = Attack(n_experiments = EXPERIMENT,training_type= training_type,target = T)
    attack.run()
                  
                            
            
            
    
    
            
            
        
        