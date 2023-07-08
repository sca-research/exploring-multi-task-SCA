
from utility import METRICS_FOLDER


import matplotlib.pyplot as plt
import numpy as np
import os
import pickle 

results = {}
results['classical'] = []
results['hierarchical'] = []
results['multi'] = []


target = 't'
#results['single_task_extracted'] = []
#results['single_task_xor_mlp'] = []




max_traces = 5
plot_n = {'flat' : 1, 'extracted' : 0, 'whole' : 2}
for k  in results.keys():
    metric = 'history_attack_on_{}_experiments_{}_1000'.format(target,k)
    file = open(METRICS_FOLDER+ metric,'rb')
    hist_dict =pickle.load(file)
    file.close()
    

 
    array_total_rank = np.empty((1000,max_traces))
    
    for i in range(1000):

        array_total_rank[i] =  hist_dict[i]['total_rank'][:max_traces]

   
    if  np.min(np.mean(array_total_rank,axis = 0)) <= 2   :

        results[k] = np.mean(array_total_rank,axis = 0)   
        
    else:
        print('error')
count = 0
handles = []
first_m = True
first_s = True

color = {'classical' : 'black','multi' : 'blue','hierarchical' : 'red'}



fig , ax = plt.subplots()
name = {'hierarchical': 'hierarchical multi-task','classical':'single-task','multi':'flat multi-task'}
for k , v in results.items():
    print(k)
    ax.set_ylim(ymin=0,ymax = 5)
    ax.set_xlim(xmin=1,xmax = max_traces+1)

    ax.set_title('Full key recovery attack using a multi target strategy')
    ax.set_ylabel('Subkey rank')  
    res = np.array(v)
    print(v.shape)
    t = np.arange(max_traces) + 1 
    print(res)
    ax.plot(t,res,label =  '{} models '.format(name[k]),color = color[k])   
    ax.set_xlabel('Traces')
    ax.set_ylabel('Full Key rank (log_2)')  

    success = len(v)
    print(k)
    print(success)
    plt.legend()
    # ax.plot(t,maxi,linestyle = 'dotted',label = 'worst multi-task model' if 'm' in k else 'worst single-task model')
  


    #ax[plot_n[tu]].fill_between(t,maxi,mini,facecolor = 'red' if 'm' in k else 'blue',alpha = 0.25)    
    count +=1

plt.yticks( [1,2,3,4,5])
plt.xticks( [1,2,3])
plt.show()