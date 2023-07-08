
from utility import METRICS_FOLDER


import matplotlib.pyplot as plt
import numpy as np
import os
import pickle 

results = {}
results['single-task-twin'] = []
results['single-task-xor'] = []
results['multi-task'] = []

#results['single_task_extracted'] = []
#results['single_task_xor_mlp'] = []




max_traces = 1000
plot_n = {'flat' : 1, 'extracted' : 0, 'whole' : 2}
for metric in os.listdir(METRICS_FOLDER):
    
    file = open(METRICS_FOLDER+ metric,'rb')
    hist_dict =pickle.load(file)
    file.close()
    
    if 'twin' in metric:
        scenario = 'single-task-twin'
    elif 'xor' in metric:
        scenario = 'single-task-xor'
    elif 'multi' in metric:
        scenario = 'multi-task'
    else:
        continue
        
    
    

        
    array_total_rank = np.empty((1000,max_traces))
    
    for i in range(1000):

        array_total_rank[i] =  hist_dict[i]['total_rank'][:max_traces]

   
    if  np.min(np.mean(array_total_rank,axis = 0)) <= 2   :
        print(metric)
        print(np.min(np.where(np.mean(array_total_rank,axis = 0) <= 2)))
        results[scenario].append(np.mean(array_total_rank,axis = 0)   )
    else:
        print('error')
count = 0
handles = []
first_m = True
first_s = True

color = {'single-task-xor' : 'black','multi-task' : 'blue','single-task-twin' : 'red'}



fig , ax = plt.subplots()

for k , v in results.items():
    if len(v) == 0:
        continue
    ax.set_ylim(ymin=0,ymax = 16)
    ax.set_xlim(xmin=1,xmax = max_traces//10)
    name = {'':'cnn','mlp':'mlp'}
    ax.set_title('Subkey recovery attack ')
    ax.set_xlabel('Traces')
    ax.set_ylabel('Subkey rank')  
    res = np.array(v)
    mu = np.mean(res,axis = 0)
    maxi = np.max(res,axis = 0)
    mini = np.min(res,axis = 0)
    t = np.arange(max_traces)
    ax.plot(t,mu,label =  'average models {}'.format(k),color = color[k])   
    ax.plot(t,mini,linestyle = 'dashed',label = 'best model {} '.format(k),color = color[k] )       

    success = len(v)
    print(k)
    print(success)
    plt.legend()
    # ax.plot(t,maxi,linestyle = 'dotted',label = 'worst multi-task model' if 'm' in k else 'worst single-task model')
  


    #ax[plot_n[tu]].fill_between(t,maxi,mini,facecolor = 'red' if 'm' in k else 'blue',alpha = 0.25)    
    count +=1



plt.show()