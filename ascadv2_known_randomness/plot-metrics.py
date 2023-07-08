from utility import METRICS_FOLDER


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import os
import pickle 

results = {}
results_100k = {}
# results['single_task_flat'] = []
results['single_task_xor_flat'] = []
results['multi_task_flat'] = []

# results['single_task_extracted'] = []
results['single_task_xor_extracted'] = []
results['multi_task_extracted'] = []
#results['single_task_softmax_check'] = []

# results['single_task_whole'] = []
results['single_task_xor_whole'] = []
results['multi_task_whole'] = []

#results_100k['single_task_xor_flat'] = []
results_100k['multi_task_flat'] = []

# results['single_task_extracted'] = []
#results_100k['single_task_xor_extracted'] = []
results_100k['multi_task_extracted'] = []
#results['single_task_softmax_check'] = []

# results['single_task_whole'] = []
#results_100k['single_task_xor_whole'] = []
results_100k['multi_task_whole'] = []



max_traces = 200
plot_n = {'flat' : 1, 'extracted' : 0, 'whole' : 2}
for metric in os.listdir(METRICS_FOLDER):
    if 'third' in metric or ('check' in metric) or ('saved' in metric):
        continue
 
    print(METRICS_FOLDER + 'saved_{}'.format(metric))
    split_metric = metric.split('_')[:-2] if not ('100000' in metric) else metric.split('_')[:-3]
    
    
    x = split_metric.pop(0)
    x = split_metric.pop(0)
    x = split_metric.pop(0)
 
    model_type = '_'.join(split_metric)
    if os.path.isfile(METRICS_FOLDER + 'saved_{}.npy'.format(metric)):
        array_total_rank = np.load(METRICS_FOLDER + 'saved_{}.npy'.format(metric))
        print('here')
        if  np.any(np.min(array_total_rank,axis = 0) <2) :
            
            if not ('100000' in metric):
                results['{}'.format(model_type)].append(array_total_rank  )
                
            else:
                results_100k['{}'.format(model_type)].append(array_total_rank )
        else:
            print('error')
    # else:
    #     file = open(METRICS_FOLDER+ metric,'rb')
    #     hist_dict =pickle.load(file)
    #     file.close()
        

               
    #     array_total_rank = np.empty((1000,max_traces))
        
    #     for i in range(1000):
    
    #         array_total_rank[i] =  hist_dict[i]['total_rank'][:max_traces]
    
    #     np.save(METRICS_FOLDER + 'saved_{}'.format(metric),np.mean(array_total_rank,axis = 0) )

        # if  np.any(np.min(array_total_rank,axis = 0) <2) :
            
        #     if not ('100000' in metric):
        #         results['{}'.format(model_type)].append(np.mean(array_total_rank,axis = 0)   )
                
        #     else:
        #         results_100k['{}'.format(model_type)].append(np.mean(array_total_rank,axis = 0)   )
        # else:
        #     print('error')
        
count = 0
handles = []


color = {'single_task_xor_flat' : 'blue','multi_task_flat' : 'green','single_task_xor_extracted' : 'red','multi_task_extracted' : 'black','single_task_xor_whole' : 'orange','multi_task_whole' : 'pink'}
color_100k = {'single_task_xor_flat' : 'blue','multi_task_flat' : 'green','single_task_xor_extracted' : 'red','multi_task_extracted' : 'black','single_task_xor_whole' : 'orange','multi_task_whole' : 'pink'}


for ty in ['extracted','whole','flat']:
    fig , ax = plt.subplots()
    
    for k , v in results.items():
        if ty not in k:
            continue
        if len(v) == 0:
            continue
        tu = k.split('_')[1]
        ax.set_ylim(ymin=0,ymax = 128)
        ax.set_xlim(xmin=1,xmax = max_traces)
        name = {'flat':'separated','extracted':'fully-extracted','whole':'concatenated'}
        ax.set_title('Full key recovery attack '+ name[ty] + ' scenario')
        ax.set_xlabel('Traces')
        ax.set_ylabel('Full Key rank (log_2)')  
        res = np.array(v)
        mu = np.mean(res,axis = 0)
        maxi = np.max(res,axis = 0)
        mini = np.min(res,axis = 0)
        t = np.arange(max_traces)
        ax.plot(t,mu, linewidth=1.5,color = 'blue' if 'multi' in k else 'black')   
        ax.plot(t,mini,linestyle = 'dashed', linewidth=1.5,color = 'blue' if 'multi' in k else 'black' )       
        print(k)
        
        success = len(v)
        print(success)
        
    for k , v in results_100k.items():
        if ty not in k:
            continue
        if len(v) == 0:
            continue
        tu = k.split('_')[1]
        ax.set_ylim(ymin=0,ymax = 128)
        ax.set_xlim(xmin=1,xmax = max_traces)
        name = {'flat':'separated','extracted':'fully-extracted','whole':'concatenated'}
        ax.set_title('Full key recovery attack '+ name[ty] + ' scenario')
        ax.set_xlabel('Traces')
        ax.set_ylabel('Full Key rank (log_2)')  
        res = np.array(v)
        mu = np.mean(res,axis = 0)
        maxi = np.max(res,axis = 0)
        mini = np.min(res,axis = 0)
        t = np.arange(max_traces)
        ax.plot(t,mu, linewidth=1.5,color = 'green' if 'multi' in k else 'grey')   
        ax.plot(t,mini,linestyle = 'dashed', linewidth=1.5,color = 'green' if 'multi' in k else 'grey' )       
        print(k)
        
        success = len(v)
        print(success)
        
    colors_elements = [     Patch(facecolor='blue', label='multi-task 200k'),
                       Patch(facecolor='black',  label='single-task-xor 200k'),
                       Patch(facecolor='green',  label='multi-task 100k'),
                       Patch(facecolor='darkgrey', label='single-task-xor 100k'),
                       Line2D([0], [0], color='black', linewidth=1.5, label='average models'),
                       Line2D([0], [0], color='black',linestyle = 'dashed', linewidth=1.5, label='best model'),
    
                       ]
    


    legend2 = plt.legend(bbox_to_anchor=(1,0.8),handles=colors_elements, loc='center right' )
    ax.add_artist(legend2)


plt.show()