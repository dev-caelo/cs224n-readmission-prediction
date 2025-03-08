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
            tokenizer=None,
            max_length=256,
            age=['anchor_age'],
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

        # FIX THIS???
        if tokenizer is None:
            raise AssertionError("Tokenizer is None in Diag")
        self.tokenizer = tokenizer
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

        # Handle special label values
        if isinstance(label, str):
            # Remove any non-numeric characters (like '+') and handle special cases
            if '+' in label:
                # Strip the '+' and any other non-numeric characters
                label = label.replace('+', '')
            # Add other special case handling as needed
            
            # Now convert to float then int
            try:
                label_int = int(float(label))
            except ValueError:
                # Fallback for completely non-numeric labels - map to a default value
                # or skip this sample by returning None
                print(f"Warning: Unable to convert label '{label}' to a number. Using default value.")
                label_int = 0  # Or some appropriate default
        else:
            label_int = int(label)

        age = np.array(item[self.age], dtype=np.float32)

        if self.others is not None:
            others = np.array(item[self.others], dtype=np.float32)
        else:
            others = 0

        inputs = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")

        if isinstance(label, str):
            label_int = int(float(label))
        else:
            label_int = int(label)

        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), age, others, label_int
        
