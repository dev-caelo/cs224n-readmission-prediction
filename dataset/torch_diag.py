import torch
import numpy as np
import pandas as pd
import csv
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class DiagTorch(Dataset):
    def __init__(
            self,
            data_path,
            label='time_until_next_admission"',
            subset='train',
            tokenizer=None,
            max_length=256,
            text_column='text',
            total_bins=2,
            columns_to_include=None,  # List of columns to include in text
            columns_to_exclude=None,  # List of columns to exclude from text
            separator=" "  # Separator between concatenated fields
    ):
       # Load data using pandas
        df = pd.read_csv(data_path)
        
        # Handle train/val/test split if needed
        if 'subset' not in df.columns:
            train_idx, temp_idx = train_test_split(df.index, test_size=0.2, random_state=42)
            val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
            df.loc[train_idx, 'subset'] = 'train'
            df.loc[val_idx, 'subset'] = 'val'
            df.loc[test_idx, 'subset'] = 'test'
        
        # Filter to the desired subset
        df = df[df['subset'] == subset]
        
        # Filter out rows with missing labels
        df = df[df[label].notna()]
        
        # Store needed info
        if tokenizer is None:
            raise AssertionError("Tokenizer is None in DiagTorch")
            
        self.tokenizer = tokenizer
        self.df = df
        self.subset = subset
        self.label = label
        self.text_column = text_column
        self.max_length = max_length
        self.total_bins = total_bins
        self.separator = separator
        
        # Determine which columns to include/exclude
        if columns_to_include is not None:
            self.columns_to_use = columns_to_include
        else:
            if columns_to_exclude is not None:
                self.columns_to_use = [col for col in df.columns if col not in columns_to_exclude and col != label and col != 'subset']
            else:
                # Use all columns except label and subset by default
                self.columns_to_use = [col for col in df.columns if col != label and col != 'subset']
        
        print(f"Using columns: {self.columns_to_use}")
        
        # For debugging in train mode
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
            if count > 0:
                print(f"Bin {bin_idx}: {count} samples")

    def bin_distribution(self, values):
        """Count how many values fall into each bin"""
        bins = np.zeros(self.total_bins, dtype=int)
        for val in values:
            bin_idx = self.bin_readmission_time(val)
            if isinstance(bin_idx, int) and bin_idx < self.total_bins:
                bins[bin_idx] += 1
        return bins
    
    def bin_readmission_time(self, days):
        """
        Bin readmission time into meaningful categories
        """
        days = int(days)
        if days <= 0:  # No readmission
            return 0
        elif 1 <= days <= 30:  # 1-30 days
            return 1
        elif 31 <= days <= 90:  # 31-90 days
            return 2
        elif 91 <= days <= 180:  # 91-180 days
            return 3
        elif 181 <= days <= 365:  # 181-365 days
            return 4
        else:  # More than 365 days
            return 5
    
    def _build_text_input(self, item):
        """
        Concatenate all selected columns into a single text string.
        """
        parts = []
        
        # Add the primary text column first if it exists
        if self.text_column in self.columns_to_use and self.text_column in item:
            parts.append(f"{self.text_column}: {str(item[self.text_column])}")
        
        # Add all other columns
        for col in self.columns_to_use:
            if col != self.text_column:  # Skip the main text column as it's already added
                # Handle NaN values
                if pd.isna(item[col]):
                    value = "unknown"
                else:
                    value = str(item[col])
                
                parts.append(f"{col}: {value}")
        
        # Join all parts with the separator
        return self.separator.join(parts)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        # Get row from dataframe
        item = self.df.iloc[i]
        
        # Build concatenated text from all relevant columns
        text = self._build_text_input(item)
        
        # Convert to lowercase if needed
        text = text.lower()
        
        # Tokenize text
        inputs = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        # Get and process label
        label = item[self.label]
        
        # Handle special label values
        if isinstance(label, str):
            if '+' in label:
                label = label.replace('+', '')
            try:
                label_float = float(label)
            except ValueError:
                print(f"Warning: Unable to convert label '{label}' to a number. Using default value.")
                label_float = 0
        else:
            label_float = float(label)
        
        # Bin the label value
        label_int = self.bin_readmission_time(label_float)
        
        # Return only the text inputs and label
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label_int, dtype=torch.long)
        }

# class DiagTorch(Dataset):
#     def __init__(
#             self,
#             data_path,
#             label='time_until_next_admission"',
#             subset='train',
#             tokenizer=None,
#             max_length=256,
#             age=['anchor_age'],
#             others=None,
#             text='text',
#             total_bins=5,
#     ):
#        # Load data using pandas - same as original Diag class
#         df = pd.read_csv(data_path)
        
#         # Handle train/val/test split if needed - same as original
#         if 'subset' not in df.columns:
#             train_idx, temp_idx = train_test_split(df.index, test_size=0.2, random_state=42)
#             val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
#             df.loc[train_idx, 'subset'] = 'train'
#             df.loc[val_idx, 'subset'] = 'val'
#             df.loc[test_idx, 'subset'] = 'test'
        
#         # Filter to the desired subset - same as original
#         df = df[df['subset'] == subset]
        
#         # Filter out rows with missing labels - same as original
#         df = df[df[label].notna()]
        
#         # Store needed info
#         if tokenizer is None:
#             raise AssertionError("Tokenizer is None in DiagTorch")
            
#         self.tokenizer = tokenizer
#         self.df = df
#         self.subset = subset
#         self.label = label
#         self.age = age
#         self.others = others
#         self.text = text
#         self.max_length = max_length
#         self.total_bins = total_bins
        
#         # For debugging in train mode
#         if subset == 'train':
#             self._print_label_stats()

#     def _print_label_stats(self):
#         """Print statistics about the distribution of labels in the dataset"""
#         labels = self.df[self.label].values
#         numeric_labels = []
        
#         for label in labels:
#             if isinstance(label, str):
#                 try:
#                     numeric_labels.append(float(label.replace('+', '')))
#                 except ValueError:
#                     print(f"Warning: Skipping non-numeric label: {label}")
#             else:
#                 numeric_labels.append(float(label))
                
#         numeric_labels = np.array(numeric_labels)
#         print(f"\nLabel statistics for {self.label}:")
#         print(f"Min: {np.min(numeric_labels)}, Max: {np.max(numeric_labels)}")
#         print(f"Mean: {np.mean(numeric_labels):.2f}, Median: {np.median(numeric_labels):.2f}")
        
#         # Print bin distribution
#         bins = self.bin_distribution(numeric_labels)
#         print("\nLabel distribution after binning:")
#         for bin_idx, count in enumerate(bins):
#             if count > 0:
#                 print(f"Bin {bin_idx}: {count} samples")

#     def bin_distribution(self, values):
#         """Count how many values fall into each bin"""
#         bins = np.zeros(self.total_bins, dtype=int)
#         for val in values:
#             bin_idx = self.bin_readmission_time(val)
#             if isinstance(bin_idx, int) and bin_idx < 181:
#                 bins[bin_idx] += 1
#         return bins
    
#     def bin_readmission_time(self, days):
#         """
#         Bin readmission time into meaningful categories
#         """
#         if days is None or days <= 0:  # No readmission recorded
#             return 0
#         elif int(days) > 0 and int(days) <= 30:
#             return 1
#         elif int(days) > 30 and int(days) <= 90:
#             return 2
#         elif int(days) > 90 and int(days) <= 180:
#             return 3
#         elif int(days) > 180 and int(days) <= 365:
#             return 4
#         else:
#             return 5

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, i):
#         # Get row from dataframe
#         item = self.df.iloc[i]

#         # Get and process text
#         text = str(item[self.text]).lower()
        
#         # Tokenize text
#         inputs = self.tokenizer(
#             text, 
#             padding='max_length', 
#             truncation=True, 
#             max_length=self.max_length, 
#             return_tensors="pt"
#         )
        
#         # Get and process label
#         label = item[self.label]
        
#         # Handle special label values
#         if isinstance(label, str):
#             if '+' in label:
#                 label = label.replace('+', '')
#             try:
#                 label_float = float(label)
#             except ValueError:
#                 print(f"Warning: Unable to convert label '{label}' to a number. Using default value.")
#                 label_float = 0
#         else:
#             label_float = float(label)
        
#         # Bin the label value
#         label_int = self.bin_readmission_time(label_float)
        
#         # Process age feature - directly from pandas
#         if isinstance(self.age, list):
#             age_values = []
#             for age_key in self.age:
#                 val = item[age_key]
#                 if pd.isna(val):
#                     age_values.append(0.0)
#                 else:
#                     age_values.append(float(val))
#             age = torch.tensor(age_values, dtype=torch.float32)
#         else:
#             val = item[self.age]
#             if pd.isna(val):
#                 age = torch.tensor([0.0], dtype=torch.float32)
#             else:
#                 age = torch.tensor([float(val)], dtype=torch.float32)
        
#         # Process other features
#         if self.others is not None:
#             if isinstance(self.others, list):
#                 other_values = []
#                 for other_key in self.others:
#                     val = item[other_key]
#                     if pd.isna(val):
#                         other_values.append(0.0)
#                     else:
#                         other_values.append(float(val))
#                 others = torch.tensor(other_values, dtype=torch.float32)
#             else:
#                 val = item[self.others]
#                 if pd.isna(val):
#                     others = torch.tensor(0.0, dtype=torch.float32)
#                 else:
#                     others = torch.tensor(float(val), dtype=torch.float32)
#         else:
#             others = torch.tensor(0.0, dtype=torch.float32)
        
#         # assert 0 <= label_int <= 180, f"Label out of bounds: {label_int} from original value {label}"
#         return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), age, others, label_int