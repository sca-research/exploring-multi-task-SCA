import argparse
import parse
import os
import numpy as np



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from multiprocessing import Process



from train_models_third_order import   model_multi_task_extracted , model_multi_task_whole, model_multi_task_flat, model_multi_task_noperm

# import dataset paths and variables
from utility import   MODEL_FOLDER
import tensorflow as tf
# import custom layers

from utility import load_dataset, load_dataset_multi_third_order ,load_model_from_name , get_rank_list_from_prob_dist , read_from_h5_file
import matplotlib.pyplot as plt

from scalib.metrics import SNR

def get_timepoints_from_snr(target,traces,labels,save_file = True,trace_values_dict = None ,classes = 256):
    n_traces, samples  = traces.shape
    print('Calculation for var : ',target)
    print(traces.shape)
    print(labels.shape)
    snr = SNR(classes,samples,1)
    traces = (traces).astype(np.int16)
    labels  = (labels).astype(np.uint16)   
    labels = labels.reshape(-1,1)
    
    snr.fit_u(traces,labels) 
    snr_val = snr.get_snr()   
    
    print('Chosen timepoint {} for target {}'.format(np.argmax(abs(snr_val[0,:])),target)) 
    return snr_val[0,:]

def get_values_model():


    (data, labels) = load_dataset_multi_third_order(n_traces = 10000,whole = False,flat = True,dataset = 'training',encoded_labels=True) 
    traces , labels_dict  = read_from_h5_file(n_traces = 10000,dataset = 'training')
    # _ , labels_dict, _ = read_from_h5_file(n_traces = 100,dataset = 'attack')
        
    # model_without_weights = cnn_unmasking(input_length = data['traces'].shape[1],input_layer = "weighted_pooling") 
    # model = load_model_unmasking_from_target(model_without_weights,target_byte,name = 'cnn_unmasking',window_type= 'first_round',input_layer = 'weighted_pooling')   
    
 
    structure = model_multi_task_whole(1,1 , [18],16 , 4,201,summary = True)                  
  

    name =  'model_multi_task_whole_all_cb1ks[18]f16ps4db1du201_third.h5'
    model = load_model_from_name(structure,name)
    # predictions = model.evaluate(data,labels,batch_size = 250)

    # rank , accuracy, prob, accuracy_top5 = get_rank_list_from_prob_dist(predictions['output_mask'],labels['output_mask'])
    # print('mask :',(np.median(rank) , accuracy, np.median(prob), accuracy_top5)) 
    to_get_layers={}
    for layer in model.layers:
       
        if 'output_rin' in layer.name:
            print(layer.name)
            to_get_layers['output_rin']  = layer.output
        if 'output_alpha' in layer.name:
            print(layer.name)
            to_get_layers['output_alpha']  = layer.output


    extractor = tf.keras.Model(inputs=model.inputs,
                        outputs=to_get_layers)
    
    
    predictions = extractor.predict(data,batch_size = 250)
    # np.save(METRICS_FOLDER + 'results_extractor_left',results[0])
    # np.save(METRICS_FOLDER + 'results_extractor_right',results[1])
    
    # intermediate = np.load(METRICS_FOLDER + 'results_extractor_left.npy')
    # mask = np.load(METRICS_FOLDER + 'results_extractor_right.npy')
    
    # intermediate = results[0]
    # mask = results[1]
    
    # softmax_intermediate = tf.keras.layers.Softmax()(intermediate)
    # softmax_mask= tf.keras.layers.Softmax()(mask)
    
    # np.save(METRICS_FOLDER + 'softmax_extractor_left',softmax_intermediate)
    # np.save(METRICS_FOLDER + 'softmax_extractor_right',softmax_mask)   
    
    # softmax_intermediate = np.load(METRICS_FOLDER + 'softmax_extractor_left.npy',allow_pickle= True)    
    # softmax_mask = np.load(METRICS_FOLDER + 'softmax_extractor_right.npy',allow_pickle= True)   
    


    # 
    # print('Byte :',(np.median(rank) , accuracy, np.median(prob), accuracy_top5))
    # argmax_alpha = np.argmax(predictions['output_alpha'],axis = 1)
    
    # # snr_intermediate = loss_snr_without_knowledge([data['traces'][:500,:,0],predictions[:500]])
    snr_rin = get_timepoints_from_snr('',traces,np.argmax(predictions['output_rin'],axis =1) )
    snr_alpha = get_timepoints_from_snr('',traces,np.argmax(predictions['output_alpha'],axis =1) )
    # rin_pred = np.argmax(predictions['output_rin'],axis =1) ^ 81
    # rin_true = np.array(labels_dict['rin'])
    # count = 0
    # for elem in range(10000):
    #     count += 1 if rin_pred[elem] == rin_true[elem] else 0
    # print(count/100)
    fig , ax = plt.subplots()
    ax.plot(snr_alpha)
    ax.plot(snr_rin)
    plt.show()

    
    # #plt.plot(snr_intermediate_from_true_values,label = 'Snr from the key')
    # plt.plot(snr_alpha,label = 'alpha')
    # plt.plot(snr_rin,label = 'rin')
    # #fig.set_title('Snr right side of the xor layer') 
    # fig.legend()

    # plt.show()
    
    
    # return metrics  


#### Training high level function
def test_model(training_type,byte,convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units,threat):
    n_traces = 5000


    flat = False
    
    whole = False
    print(flat)
    X_profiling , validation_data = load_dataset_multi_third_order(flat = flat,whole = whole,noperm = True,n_traces = n_traces,dataset = 'attack',encoded_labels=True) 
    model_t = 'model_multi_task_{}'.format(training_type)
    
    

  

    structure = model_multi_task_noperm(convolution_blocks,dense_blocks , kernel_size,filters , pooling_size,dense_units,summary = False)                  
           


    name =  '{}_{}_cb{}ks{}f{}ps{}db{}du{}_third.h5'.format(model_t,byte,convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units)
    model = load_model_from_name(structure,name)
    
    print('here')
    predictions =  model.evaluate(X_profiling,validation_data,verbose = 1, batch_size=256)
    # for k , v in predictions.items():
    #     labels = validation_data[k]
    
    #     ranks , acc, scores , acc_5 = get_rank_list_from_prob_dist(v,labels)
    #     print('=========================')
    #     print(name)
    #     print('Mean rank {}, Median rank {}, Mean score {}, Median score {}, Accuracy {}'.format(np.mean(ranks), np.median(ranks),np.mean(scores),np.median(scores),acc))
    
    # np.save(METRICS_FOLDER + 'results_{}'.format(name),np.array([np.mean(ranks), np.median(ranks),np.mean(scores),np.median(scores),acc]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--SINGLE_TASK', action="store_true", dest="SINGLE_TASK",
                        help='Classical training of the intermediates', default=False)
    parser.add_argument('--SINGLE_TASK_XOR',   action="store_true", dest="SINGLE_TASK_XOR", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI',   action="store_true", dest="MULTI", help='Adding the masks to the labels', default=True)
    parser.add_argument('--ALL',   action="store_true", dest="ALL", help='Adding the masks to the labels', default=False)
        
    args            = parser.parse_args()
  

    SINGLE_TASK        = args.SINGLE_TASK
    SINGLE_TASK_XOR        = args.SINGLE_TASK_XOR
    MULTI = args.MULTI
    ALL = args.ALL

    TARGETS = {}
    if SINGLE_TASK:   
       training_types = ['single_task']
    elif SINGLE_TASK_XOR:
        training_types = ['single_task_xor']

    elif MULTI:
        training_types = ['multi_task_noperm']
    elif ALL:
        training_types = ['single_task_xor','multi_task']
    else:
        print('No training mode selected')

    
    for model_name in os.listdir(MODEL_FOLDER):
        if 'first_batch' in model_name :
            continue

        multi_task =  'multi_task_noperm' in model_name and ( '_third' in model_name)
        if not multi_task:
            continue
        
        
        format_string = 'model_multi_task_{}_{}_cb{}ks{}f{}ps{}db{}du{}_third.h5' 
        parsed = parse.parse(format_string,model_name)
        training_type = parsed[0]
        byte = int(parsed[1]) if not parsed[1] == 'all' else 'all'
        convolution_blocks = int(parsed[2])
        kernel_size_list = parsed[3][1:-1]
    

        kernel_size_list = kernel_size_list.split(',')   
        kernel_size = [int(elem) for elem in kernel_size_list]

        filters = int(parsed[4])
        if  (not  filters == 16) :
            continue
        print(model_name)
        pooling_size = int(parsed[5])
        dense_blocks = int(parsed[6])
        dense_units = int(parsed[7])
        test_model(training_type,byte,convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units,False)

        process_eval = Process(target=test_model, args=(training_type,byte,convolution_blocks , kernel_size,filters , pooling_size,dense_blocks,dense_units,False))
        process_eval.start()
        process_eval.join()
        
                            


    print("$ Done !")