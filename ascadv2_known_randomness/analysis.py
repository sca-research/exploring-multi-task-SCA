# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:23:17 2023

@author: martho
"""

import numpy as np
from scalib.metrics import SNR
from utility import DATASET_FOLDER, TRACES_FOLDER, REALVALUES_FOLDER , read_from_h5_file
import matplotlib.pyplot as plt
from utils.generate_intermediate_values import save_real_values
import pickle

def get_snr(traces,labels,classes = 256):
    traces = traces.astype(np.int16)
    labels = labels.astype(np.uint16)
    if len(labels.shape) == 1 :
        labels = labels.reshape(-1,1)
    snr = SNR(classes,traces.shape[1],labels.shape[1])
    snr.fit_u(traces,labels)
    return snr.get_snr()

n_traces  = 200000



traces, labels_dict = read_from_h5_file(n_traces = n_traces ,dataset = 'training')




for target in ['s1^m']:
    snr = get_snr(traces,np.array(labels_dict[target])[:n_traces])
    
    # order = np.argsort(snr[0])[-200:]
    
    #
    count = 0
    for elem  in snr:
        plt.plot(elem,label = count)
        count += 1
plt.legend()  
plt.show()





