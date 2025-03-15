# Readmission Prediction Model
**Authors: [Carlo Dino](cdino@stanford.edu), [Shriya Reddy](reddysh@stanford.edu), [Yu Han Daisy Wang](daisywyh@stanford.edu)**

In this project, we assess the capabilities of the RoBERTA model with data fusion techniques presented in [Ye et. al. (2024)](https://pubmed.ncbi.nlm.nih.gov/38827058/) for
the task of readmission prediction using EHR data. We utilize the [MIMIC-IV dataset](https://physionet.org/content/mimiciv/3.1/) to souce EHR data for training and testing. We compare
our approach with the LACE index as a baseline.

## Running
'train.py' is our primary run file for running and training hybrid_fusion models and retrieving evaluation metrics. The model definition and files are stored in 'models/hybrid_fusion'.
'models/bert_finetune' contains codefile and train file for finetuning bert and running inference.
'models/LACE' contains the codefile that creates predictions based on LACE scores
'models/xgboost' contains notebooks and codefiles that run our xgboost model

TO RUN TRAIN.PY (Hybrid Fusion): Please follow this format -> 'python3 train.py ['roberta' || 'clinical_bert'] [2 || 6]'
- We choose between two BERT modules to extract textual embeddings: RoBERTa and Clinical_BERT
- We choose between two evaluation tasks: LACE (Binary classification) and 6-Bin (described in paper)

## Data
We source our dataset from MIMIC-IV, and run a number of our models on processed versions of dataframes / embeddings that are derived from MIMIC-IV data. Due to the credentialing requirement of MIMIC-IV, we cannot share these here. Instead, we provide the data parsing functionality in 'dataset/' to recreate our structure once the MIMIC-IV dataset is procured locally. Since MIMIC-IV documentation is public, we maintain the column names in our codebase to allow for reproducability.

## Project Notes
- Please see this [doc](https://docs.google.com/document/d/1k49T_YHEq8YNn3q_Eocalz-l9brFbkvZf66Ba4l1JI4/edit?usp=sharing) for more detailed calculations of LACE using MIMIC-IV.
- 
