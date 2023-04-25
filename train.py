import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import argparse
import random
random.seed(577)

import numpy as np
np.random.seed(577)
import warnings
import torch
# from GPUtil import showUtilization as gpu_usage
# from numba import cuda

# def free_gpu_cache():
#     print("Initial GPU Usage")
#     gpu_usage()                             

#     torch.cuda.empty_cache()

#     cuda.select_device(0)
#     cuda.close()
#     cuda.select_device(0)

#     print("GPU Usage after emptying the cache")
#     gpu_usage()

# free_gpu_cache() 
# import gc
# gc.collect()
# torch.cuda.empty_cache()
# torch.cuda.empty_cache()

warnings.filterwarnings("ignore", category=UserWarning)

import torch
torch.set_default_tensor_type(torch.FloatTensor)
torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
torch_device = torch.device("mps")
from sklearn.model_selection import KFold

'''
NOTE: Do not change any of the statements above regarding random/numpy/pytorch.
You can import other built-in libraries (e.g. collections) or pre-specified external libraries
such as pandas, nltk and gensim below. 
Also, if you'd like to declare some helper functions, please do so in utils.py and
change the last import statement below.
'''

import gensim.downloader as api

from neural_archs import LSTM, LSTM_double_fc, BERT, BERT_double_fc
from utils import WiCDataset
import pandas as pd
from sklearn.model_selection import KFold

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', choices=['lstm', 'bert'], default='lstm', type=str)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--hidden_nodes', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.00001, type=float)
    parser.add_argument('--version', choices=['baseline', 'separate_class', 'joint_class'], default='baseline', type=str)
    parser.add_argument('--attribute', choices=['sentiment', 'topic', 'source'], default='sentiment', type=str)

    args = parser.parse_args()
    print(args)

    glove_embs = api.load("glove-wiki-gigaword-50")
    glove_embs.add_vector('UNK', np.zeros(50))

    epochs = args.epochs

    train_dataset = WiCDataset()
    # train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 1)

    loss_fn = torch.nn.CrossEntropyLoss()

    if args.version == 'baseline':
        fn = lambda x: x['one_hot_bias']
    elif args.version == 'separate_class':
        if args.attribute == 'sentiment':
            fn = lambda x: torch.cat([x['one_hot_bias'], x['one_hot_sentiment']], axis = 1)
        elif args.attribute == 'topic':
            fn = lambda x: torch.cat([x['one_hot_bias'], x['one_hot_topic']], axis = 1)
        else:
            fn = lambda x: torch.cat([x['one_hot_bias'], x['one_hot_source']], axis = 1)
    elif args.version == 'joint_class':
        if args.attribute == 'sentiment':
            fn = lambda x: x['one_hot_bias_sentiment']
        elif args.attribute == 'topic':
            fn = lambda x: x['one_hot_bias_topic']
        else:
            fn = lambda x: x['one_hot_bias_source']

    kf = KFold(n_splits=5)
    

    
    validation_accuracies1 = []
    validation_accuracies2 = []
    validation_losses = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, sampler=train_sampler)
        val_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=val_sampler)
        
        if args.model == 'lstm': 
            if args.version == 'separate_class':
                model = LSTM_double_fc(glove_embs, args.version, args.attribute, args.hidden_nodes).to(torch_device)
            else:
                model = LSTM(glove_embs, args.version, args.attribute, args.hidden_nodes).to(torch_device)
        else:
            if args.version == 'separate_class':
                model = BERT_double_fc(glove_embs, args.version, args.attribute).to(torch_device)
            else:
                model = BERT(glove_embs, args.version, args.attribute).to(torch_device)


        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        for epoch in range(epochs):
            print(epoch)
            for data in train_dataset_loader:
                pred = model(data)
                loss = loss_fn(pred.to(torch_device), fn(data).to(torch_device))
                loss.backward()
                optimizer.step()

        if args.version == 'separate_class':
            acc1 = acc2 = loss_val = 0
            for data in val_dataset_loader:
                pred = model(data)
                loss_val += loss_fn(pred.to(torch_device), fn(data).to(torch_device)).item()
                result = fn(data.to(torch_device))
                if torch.argmax(pred[:,:3]) == torch.argmax(result[:,:3]):
                    acc1+=1
                if torch.argmax(pred[:,3:]) == torch.argmax(result[:,3:]):
                    acc2+=1
            validation_accuracies1.append(acc1/len(val_dataset_loader))
            validation_accuracies2.append(acc2/len(val_dataset_loader))
            validation_losses.append(loss_val/len(val_dataset_loader))
        else:
            acc = 0
            loss_val = 0
            for data in val_dataset_loader:
                pred = model(data)
                loss_val += loss_fn(pred.to(torch_device), fn(data).to(torch_device)).item()
                if torch.argmax(pred.to(torch_device)) == torch.argmax(fn(data).to(torch_device)):
                    acc+=1
            validation_accuracies1.append(acc/len(val_dataset_loader))
            validation_losses.append(loss_val/len(val_dataset_loader))
            
        print("val1=",validation_accuracies1,"val2 =",validation_accuracies2,"loss =",validation_losses)
        break
    print(np.mean(validation_accuracies1), np.mean(validation_accuracies2), np.mean(validation_losses))
    with open('vishal_result.txt', 'w') as result_file:
        result_file.write(f"{np.mean(validation_accuracies1)} {np.mean(validation_accuracies2)} {np.mean(validation_losses)}\n")
# hyperparameters = epochs, no of layers, learning rate, regularization param

# Output - bias_accuracy sentiment_accuracy loss
