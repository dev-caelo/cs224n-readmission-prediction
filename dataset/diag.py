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

        # Print label statistics for debugging
        if subset == 'train':
            self._print_label_stats()

    def _print_label_stats(self):
        """Print statistics about the distribution of labels in the dataset"""
        labels = self.df[self.label].values
        numeric_labels = []
        
        for label in labels:
            if isinstance(label, str):
                try:
                    numeric_labels.append(float(label.replace('+', '')))
                except ValueError:
                    print(f"Warning: Skipping non-numeric label: {label}")
            else:
                numeric_labels.append(float(label))
                
        numeric_labels = np.array(numeric_labels)
        print(f"\nLabel statistics for {self.label}:")
        print(f"Min: {np.min(numeric_labels)}, Max: {np.max(numeric_labels)}")
        print(f"Mean: {np.mean(numeric_labels):.2f}, Median: {np.median(numeric_labels):.2f}")
        
        # Print bin distribution
        bins = self.bin_distribution(numeric_labels)
        print("\nLabel distribution after binning:")
        for bin_idx, count in enumerate(bins):
            print(f"Bin {bin_idx}: {count} samples")

    def bin_distribution(self, values):
        """Count how many values fall into each bin"""
        bins = np.zeros (2, dtype=int)  # 28 classes
        for val in values:
            bin_idx = self.bin_readmission_time(val)
            if bin_idx < 2:
                bins[bin_idx] += 1
        return bins
    
    def bin_readmission_time(self, days):
        """
        Bin readmission time into meaningful categories
        """
        if int(days) < 30 and int(days) > -1:  # Within 180 days 
            return 1
        else:  # More than 6 months or no readmission
            return 0

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
                label_float = int(float(label))
            except ValueError:
                # Fallback for completely non-numeric labels - map to a default value
                # or skip this sample by returning None
                print(f"Warning: Unable to convert label '{label}' to a number. Using default value.")
                label_float = 0  # Or some appropriate default
        else:
            label_int = int(label)
                # Bin the label value into appropriate class (0-27)
        
        label_int = self.bin_readmission_time(label_float)

        # Process age feature
        age = np.array(item[self.age], dtype=np.float32)

        age = np.array(item[self.age], dtype=np.float32)

        if self.others is not None:
            others = np.array(item[self.others], dtype=np.float32)
        else:
            others = 0

        inputs = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")

        assert 0 <= label_int <= 180, f"Label out of bounds: {label_int} from original value {label}"
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), age, others, label_int
