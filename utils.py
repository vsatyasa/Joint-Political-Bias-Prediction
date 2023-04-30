from torch.utils.data import Dataset
import pandas as pd
import nltk
import torch
import nltk
import numpy as np
import torch.nn as nn
import torch.optim as optim
import ast
from collections import Counter
import gensim.downloader as api
from nltk.corpus import words
from torch.utils.data import DataLoader, Dataset, random_split
import os
from ast import literal_eval
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

NLTK_STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", 
                   "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 
                   'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
                   'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
                   'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 
                   'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 
                   'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 
                   'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
                   'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 
                   'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 
                   'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ma']


class Vocab():
    vocab = {} 
    reverse_vocab = {}
    
    def __init__(self, glove_embs, data_set_frame):
        self.glove_embs = glove_embs
        self.data_set_frame = data_set_frame
        self.load_vocab()
        # self.append_target = append_target

    def get_size(self):
        return len(self.vocab) + 2
        
    def load_vocab(self):
        self.vocab = {word: idx for idx, word in enumerate(self.glove_embs.index_to_key)}
        
        ## build reverse vocab
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        
    def get_word(self, idx):
        return self.reverse_vocab.get(idx, "UNK")

    def get_idx(self, token):
        return self.vocab.get(token, len(self.vocab)-1)

    # Used to Representation of a sentence / context
    def build_idx_sequence(self, tokens):
        seq =  [self.get_idx(token) for token in tokens]
        return seq + [self.get_idx("UNK") for x in range(484-len(seq))]

class WiCDataset(Dataset):

    lemmatizer = nltk.stem.WordNetLemmatizer()
    stemmer = nltk.stem.PorterStemmer()

    def __init__(self):
        
        if os.path.exists('preprocessed.csv'):
            self.text_data = pd.read_csv('preprocessed.csv')
            
            self.text_data['glove_encoded_content'] = self.text_data['glove_encoded_content'].apply(literal_eval)
            self.text_data['one_hot_topic'] = self.text_data['one_hot_topic'].apply(literal_eval)
            self.text_data['one_hot_source'] = self.text_data['one_hot_source'].apply(literal_eval)
            self.text_data['one_hot_bias'] = self.text_data['one_hot_bias'].apply(literal_eval)
            self.text_data['one_hot_sentiment'] = self.text_data['one_hot_sentiment'].apply(literal_eval)
            self.text_data['one_hot_bias_sentiment'] = self.text_data['one_hot_bias_sentiment'].apply(literal_eval)
        else:
            self.text_data = pd.read_csv('sentiment_labels.csv')
            self.glove_file = api.load("glove-wiki-gigaword-50")
            self.glove_file.add_vector('UNK', np.zeros(50))
            
            ## Build Tokens for each context sentence
            self.text_data['_content'] = self.text_data['content'].apply(self.preprocess)

            self.vocab = Vocab(self.glove_file, self.text_data)
            self.text_data['glove_encoded_content'] = self.text_data['_content'].apply(self.vocab.build_idx_sequence)
            self.text_data['one_hot_topic'] = pd.get_dummies(self.text_data['topic']).values.tolist()
            self.text_data['one_hot_source'] = pd.get_dummies(self.text_data['source']).values.tolist()
            self.text_data['one_hot_bias'] = pd.get_dummies(self.text_data['bias']).values.tolist()
            self.text_data['one_hot_sentiment'] = pd.get_dummies(self.text_data['final_label']).values.tolist()
            self.text_data['one_hot_bias_sentiment'] = pd.get_dummies(self.text_data['final_label'].astype(str) + self.text_data['bias'].astype(str)).values.tolist()
            self.text_data['one_hot_bias_topic'] = pd.get_dummies(self.text_data['topic'].astype(str) + self.text_data['bias'].astype(str)).values.tolist()
            self.text_data['one_hot_bias_source'] = pd.get_dummies(self.text_data['source'].astype(str) + self.text_data['bias'].astype(str)).values.tolist()
            self.text_data[['one_hot_bias_sentiment',
                            'one_hot_bias_topic',
                            'one_hot_bias_source',
                            'one_hot_sentiment',
                            'one_hot_bias',
                            'one_hot_source',
                            'one_hot_topic',
                            'glove_encoded_content',
                            'content']].to_csv('preprocessed.csv')
        self.text_data['bert_encoded_content'] = self.text_data['content'].apply(lambda text: bert_tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt"))
        

    def get_vocab(self):
        return self.vocab
        
    def preprocess(self, text):
        
        ## PreProcess Flags
        STOP_WORDS_REMOVAL = True
        LEMMATIZE = True
        STEM = False
        
        tokens = nltk.word_tokenize(text)
        pre_processed_tokens = []
        
        for token in tokens:
            if token in NLTK_STOP_WORDS and STOP_WORDS_REMOVAL:
                continue
            
            token = token.lower()
            
            ##lemmatize
            if LEMMATIZE:
                token = self.lemmatizer.lemmatize(token)
            
            ## stem the token
            if STEM:
                token = self.stemmer.stem(token)
            
            pre_processed_tokens.append(token)
        
        return pre_processed_tokens

    def __len__(self):
        return self.text_data.shape[0]
    
    ## Vocab size is calculated by considering the padding and unknown tokens
    def get_vocab_size(self):
        self.vocab.get_size()
        
    def get_label_representation(self, label):
        return 0 if label == 'F' else 1
    
    def get_pos_label(self, idx):
        if self.text_data.iloc[idx, 1] == 'N':
            return 0
        return 1 
    
    
    def __getitem__(self, idx):
        numeric_data = ast.literal_eval(self.text_data['one_hot_bias_topic'].iloc[idx])
        numeric_data_source = ast.literal_eval(self.text_data['one_hot_bias_source'].iloc[idx])

        return {
            'glove_encoded_content': torch.tensor(self.text_data['glove_encoded_content'].iloc[idx]),
            'one_hot_topic': torch.tensor(self.text_data['one_hot_topic'].iloc[idx]),
            'one_hot_source': torch.tensor(self.text_data['one_hot_source'].iloc[idx]),
            'one_hot_bias': torch.FloatTensor(self.text_data['one_hot_bias'].iloc[idx]),
            'one_hot_sentiment': torch.tensor(self.text_data['one_hot_sentiment'].iloc[idx]),
            'one_hot_bias_sentiment': torch.FloatTensor(self.text_data['one_hot_bias_sentiment'].iloc[idx]),
             
            'one_hot_bias_topic': torch.FloatTensor(numeric_data),
            'one_hot_bias_source': torch.FloatTensor(numeric_data_source),
            'bert_encoded_content': self.text_data['bert_encoded_content'].iloc[idx]
        }
