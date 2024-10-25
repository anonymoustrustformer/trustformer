# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 08:59:59 2024

@author: aliab
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

def load_dataset_2(name='wmt19', n_records_train=200, n_records_test=10, train_offset = 0, test_offset = 0, direction = 'ru_to_en'):
    print("Loading dataset:", name)
    translation_dataset = load_dataset('wmt19', 'ru-en')
    print("Dataset loaded successfully")
    train_data = translation_dataset['train'][train_offset:n_records_train]
    valid_data = translation_dataset['validation'][test_offset:n_records_test]
    if  direction == 'ru_to_en':
        source_data = [train_data['translation'][i]['ru'] for i in range(len(train_data['translation']))]
        target_data = [train_data['translation'][i]['en'] for i in range(len(train_data['translation']))]
        source_data_validation = [valid_data['translation'][i]['ru'] for i in range(len(valid_data['translation']))]
        target_data_validation = [valid_data['translation'][i]['en'] for i in range(len(valid_data['translation']))]
    elif direction == 'en_to_ru':
        source_data = [train_data['translation'][i]['en'] for i in range(len(train_data['translation']))]
        target_data = [train_data['translation'][i]['ru'] for i in range(len(train_data['translation']))]
        source_data_validation = [valid_data['translation'][i]['en'] for i in range(len(valid_data['translation']))]
        target_data_validation = [valid_data['translation'][i]['ru'] for i in range(len(valid_data['translation']))]
        

    return source_data, target_data, source_data_validation, target_data_validation



def tokenized_dataset(name='wmt19', n_records_train = 200, n_records_test = 10, max_seq_length = 100, train_offset = 0, test_offset = 0):
    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    # tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    # encoded = tokenizer(src_text, return_tensors="pt", src_lang=src_lang)
    source_data, target_data, source_data_validation, target_data_validation = load_dataset_2(name='wmt19',n_records_train=n_records_train,n_records_test=n_records_test, train_offset = train_offset, test_offset = test_offset)
    source_tokens = tokenizer(source_data, return_tensors="pt", truncation=True, padding=True,max_length=max_seq_length)
    target_tokens = tokenizer(target_data, return_tensors="pt", truncation=True, padding=True, max_length=max_seq_length)
    source_tokens_validation = tokenizer(source_data_validation, return_tensors="pt", truncation=True, padding=True,max_length=max_seq_length)
    target_tokens_validation = tokenizer(target_data_validation, return_tensors="pt", truncation=True, padding=True, max_length=max_seq_length)

    source_ids = source_tokens['input_ids'].tolist()
    target_ids= target_tokens['input_ids'].tolist()

    source_ids_validation = source_tokens_validation['input_ids'].tolist()
    target_ids_validation = target_tokens_validation['input_ids'].tolist()    
    
    return {'source_ids':source_ids, 
            'target_ids':target_ids, 
            'source_ids_validation':source_ids_validation, 
            'target_ids_validation':target_ids_validation}

def decode_tokens(tokens):
    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    # tokenizer = MBart50TokenizerFast.from_pretrained(model_name="facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    generated_tokens = [tokenizer.decode(token_id) for token_id in tokens]
    return generated_tokens
    