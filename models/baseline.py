"""
baseline.py

This file contains our project's baseline, the LACE index. The LACE index is a commonly
used metric to predict the chances of hospital readmission. The index buckets patients
into the following: {Low, Moderate, High}.

LACE expects the data to be cleaned and parsed as lists/dicts, and will output its score per subject_id
in a pandas Dataframe.
"""

from typing import Iterable
import pandas as pd

class LACE():
  def __init__(self, 
               patient_count : int,
               subject_ids : Iterable[int],
               hadm_ids : Iterable[int],
               admit_data : Iterable,
               comorbidity_scores : dict,
               ed_visit_counts : dict
               ):
      
      # Store necessary LACE information
      self.patient_count = patient_count
      self.subject_ids = subject_ids
      self.hadm_ids = hadm_ids
      self.admit_data = admit_data
      self.comorbidity_scores = comorbidity_scores
      self.ed_visit_counts = ed_visit_counts
  
  def forward(self):
    # Initialize list for LACE scores
    lace_scores = []

    for i in range(self.patient_count):
        subject_id = self.subject_ids[i]
        hadm_id = self.hadm_ids[i]
        
        # Calculate L - Length of stay
        los_days = self.admit_data[i]['los_days']
        l_score = 0
        if los_days == 1:
            l_score = 1
        elif los_days == 2:
            l_score = 2
        elif los_days == 3:
            l_score = 3
        elif 4 <= los_days <= 6:
            l_score = 4
        elif 7 <= los_days <= 13:
            l_score = 5
        else:  # 14 or more
            l_score = 7
        
        # Calculate A - Acuity of admission
        a_score = 3 if self.admit_data[i]['admission_type'] == 'EMERGENCY' else 0
        
        # Calculate C - Comorbidity
        c_score = self.comorbidity_scores[subject_id]['final_c_score']
        
        # Calculate E - ED visits
        e_score = self.ed_visit_counts.get(subject_id, 0)
        
        # Calculate total LACE score
        total_lace = l_score + a_score + c_score + e_score
        
        lace_scores.append({
            'subject_id': subject_id,
            'hadm_id': hadm_id,
            'l_score': l_score,
            'a_score': a_score,
            'c_score': c_score,
            'e_score': e_score,
            'lace_total': total_lace,
            'readmission_risk': 'High' if total_lace >= 10 else 'Moderate' if total_lace >= 5 else 'Low'
        })

    # Create and return LACE scores DataFrame
    lace_scores_df = pd.DataFrame(lace_scores)
    return lace_scores_df