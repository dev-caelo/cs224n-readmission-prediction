# Forked and edited from original authors: https://github.com/haidog-yaqub/Clinical_HybridFusion/blob/main/dataset/diag.py

import pandas as pd
import numpy as np
import re
import torch
import random
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


from transformers import AutoTokenizer, AutoModelForMaskedLM


class Diag(Dataset):
    def __init__(
            self,
            df,
            label='Diagnosis_new',
            subset='train',
            options_name='bert-base-uncased',
            max_length=256,
            age=['Age_new'],
            others=None,
            # others=['Sex', 'Fire_Involvement'],
            text='text',
    ):
        df = pd.read_csv(df)

        if 'subset' not in df.columns:
            # For example, split 80% train, 10% val, 10% test
            train_idx, temp_idx = train_test_split(df.index, test_size=0.2, random_state=42)
            val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
            df['subset'] = 'train'
            df.loc[val_idx, 'subset'] = 'val'
            df.loc[test_idx, 'subset'] = 'test'

        df = df[df['subset'] == subset]
        df = df[df[label].notna()]

        # if 'Location' in others:
        #     for i in range(9):
        #         others.append('Location_'+str(i))
        #     others.remove('Location')
        #
        # if 'Body_Part_new' in others:
        #     for i in range(25):
        #         others.append('Body_Part_'+str(i))
        #     others.remove('Body_Part_new')
        self.tokenizer = AutoTokenizer.from_pretrained(options_name, model_max_length=max_length)
        # self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', options_name)
        self.df = df
        self.subset = subset
        self.label = label
        self.age = age
        self.others = others
        self.text = text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        item = self.df.iloc[i]

        text = str(item[self.text])
        text = text.lower()
        label = item[self.label]

        age = np.array(item[self.age], dtype=np.float32)

        if self.others is not None:
            others = np.array(item[self.others], dtype=np.float32)
        else:
            others = 0

        inputs = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")

        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), age, others, int(label)
