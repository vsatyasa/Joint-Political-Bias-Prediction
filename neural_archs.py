import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import nltk
from nltk.corpus import wordnet
from transformers import BertModel

class LSTM(torch.nn.Module):
    # Input - encoded sentence 
    # Output - bias
    def __init__(self, embedding, version, attribute, n_hiddel_layer = 100):
        super(LSTM, self).__init__()
        self.version = version
        self.embedding = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embedding.vectors)).to(torch.device('cuda'))

        sentence_length = 484
        self.lstm = torch.nn.LSTM(50*sentence_length, n_hiddel_layer)
        
        if version == 'baseline':
            self.fc1 = torch.nn.Linear(n_hiddel_layer, 3)
        elif version == 'joint_class':
            if attribute == 'source':
                self.fc1 = torch.nn.Linear(n_hiddel_layer, 149)
            elif attribute == 'topic':
                self.fc1 = torch.nn.Linear(n_hiddel_layer, 291)
            else:
                self.fc1 = torch.nn.Linear(n_hiddel_layer, 6)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # TODO: Implement LSTM forward pass
        ip = x['glove_encoded_content']
        o_embedding = self.embedding(x['glove_encoded_content'].to(torch.device("cuda"))).view((len(ip),-1))
        lstm, _ = self.lstm(o_embedding)
        relu = self.relu(lstm)
        fc1 = self.fc1(relu)
        return self.softmax(fc1)
    
    
class LSTM_double_fc(torch.nn.Module):
    # Input - encoded sentence 
    # Output - bias
    def __init__(self, embedding, version, n_hiddel_layer, attribute):
        super(LSTM_double_fc, self).__init__()
            
        self.version = version
        
        self.embedding = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embedding.vectors)).to(torch.device('cuda'))
        sentence_length = 484
        self.lstm = torch.nn.LSTM(50*sentence_length, n_hiddel_layer)
        
        self.fc1 = torch.nn.Linear(n_hiddel_layer, 3)
        if attribute == 'source':
            self.fc2 = torch.nn.Linear(n_hiddel_layer, 149)
        elif attribute == 'topic':
            self.fc2 = torch.nn.Linear(n_hiddel_layer, 105)
        else:
            self.fc2 = torch.nn.Linear(n_hiddel_layer, 2)
        
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # TODO: Implement LSTM forward pass
        ip = x['glove_encoded_content']
        o_embedding = self.embedding(x['glove_encoded_content']).to(torch.device('cuda')).view((len(ip),-1))
        lstm, _ = self.lstm(o_embedding)
        relu = self.relu(lstm)
        fc1 = self.fc1(relu)
        fc2 = self.fc2(relu)
        return torch.cat([self.softmax(fc1), self.softmax(fc2)], axis = 1)
    
    
class BERT(torch.nn.Module):
    # Input - encoded sentence 
    # Output - bias
    def __init__(self, embedding, version, attribute):
        super(BERT, self).__init__()
            
        self.version = version
        
        # Bert's o/p is 768
        n_hiddel_layer = 768
        
        self.bert = BertModel.from_pretrained('bert-base-cased')
        
        if version == 'baseline':
            self.fc1 = torch.nn.Linear(n_hiddel_layer, 3)
        elif version == 'joint_class':
            if attribute == 'source':
                self.fc1 = torch.nn.Linear(n_hiddel_layer, 149)
            elif attribute == 'topic':
                self.fc1 = torch.nn.Linear(n_hiddel_layer, 291)
            else:
                self.fc1 = torch.nn.Linear(n_hiddel_layer, 6)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # TODO: Implement LSTM forward pass
        _, bert = self.bert(input_ids= x['bert_encoded_content']['input_ids'].view(-1, x['bert_encoded_content']['input_ids'].shape[-1]).to(torch.device('cuda')), attention_mask=x['bert_encoded_content']['attention_mask'].view(-1, x['bert_encoded_content']['attention_mask'].shape[-1]).to(torch.device('cuda')),return_dict=False)

        relu = self.relu(bert)
        fc1 = self.fc1(relu)
        return self.softmax(fc1)
    
class BERT_double_fc(torch.nn.Module):
    # Input - encoded sentence 
    # Output - bias
    def __init__(self, embedding, version, attribute):
        super(BERT_double_fc, self).__init__()
            
        self.version = version
        
        n_hiddel_layer = 768
        
        self.bert = BertModel.from_pretrained('bert-base-cased')
        
        self.fc1 = torch.nn.Linear(n_hiddel_layer, 3)
        if attribute == 'source':
            self.fc2 = torch.nn.Linear(n_hiddel_layer, 149)
        elif attribute == 'topic':
            self.fc2 = torch.nn.Linear(n_hiddel_layer, 105)
        else:
            self.fc2 = torch.nn.Linear(n_hiddel_layer, 2)
        
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # TODO: Implement LSTM forward pass
        
        _, bert = self.bert(input_ids= x['bert_encoded_content']['input_ids'].view(-1, x['bert_encoded_content']['input_ids'].shape[-1]).to(torch.device('cuda')), attention_mask=x['bert_encoded_content']['attention_mask'].view(-1, x['bert_encoded_content']['attention_mask'].shape[-1]).to(torch.device('cuda')),return_dict=False)

        relu = self.relu(bert)
        fc1 = self.fc1(relu)
        fc2 = self.fc2(relu)
        return torch.cat([self.softmax(fc1), self.softmax(fc2)], axis = 1)
