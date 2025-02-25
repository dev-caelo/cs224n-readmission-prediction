import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)

# Create 10 fake patients
patient_count = 10000

# Generate patient IDs and admission IDs
subject_ids = list(range(10001, 10001 + patient_count))
hadm_ids = list(range(20001, 20001 + patient_count))

# Create a base date to work from
base_date = datetime(2023, 1, 1)

# Function to generate random dates
def random_date(start_date, range_days):
    return start_date + timedelta(days=random.randint(0, range_days))

# ICD10 codes for comorbidities based on the document
icd10_categories = {
    'Previous myocardial infarction': ['I21.3', 'I22.0', 'I25.2'],
    'Cerebrovascular disease': ['I60.9', 'I63.4', 'G45.9'],
    'Peripheral vascular disease': ['I70.2', 'I73.9', 'Z95.8'],
    'Diabetes without complications': ['E11.9', 'E10.9', 'E14.0'],
    'Congestive heart failure': ['I50.0', 'I50.9', 'I11.0'],
    'Diabetes with end organ damage': ['E11.2', 'E10.3', 'E11.5'],
    'Chronic pulmonary disease': ['J44.9', 'J45.9', 'J47'],
    'Mild liver disease': ['K76.0', 'K70.0', 'B18.1'],
    'Any tumor': ['C50.9', 'C34.1', 'C61'],
    'Dementia': ['F03', 'G30.9', 'F05.1'],
    'Connective tissue disease': ['M05.9', 'M06.9', 'M32.9'],
    'AIDS': ['B20', 'B23.8'],
    'Moderate or severe liver disease': ['K70.4', 'K72.9', 'I85.0'],
    'Metastatic solid tumor': ['C78.7', 'C79.5']
}

# -------------------- 1. Create admissions table -------------------- #

# Initialize lists for admissions data
admit_data = []

for i in range(patient_count):
    # Generate random admission date
    admit_date = random_date(base_date, 180)
    
    # Determine length of stay (varies between 1-20 days)
    los_days = np.random.choice([1, 2, 3, np.random.randint(4, 7), 
                                  np.random.randint(7, 14), np.random.randint(14, 21)],
                                p=[0.1, 0.15, 0.15, 0.3, 0.2, 0.1])
    
    # Calculate discharge date
    discharge_date = admit_date + timedelta(days=int(los_days))
    
    # Determine if admitted through ED (70% probability)
    admitted_through_ed = np.random.choice([True, False], p=[0.7, 0.3])
    
    # Generate ED admission times if applicable
    if admitted_through_ed:
        ed_reg_time = admit_date - timedelta(hours=np.random.randint(1, 12))
        ed_out_time = admit_date
    else:
        ed_reg_time = None
        ed_out_time = None
    
    # All patients are discharged alive for LACE calculation purposes
    death_time = None
    
    # Add to admissions data
    admit_data.append({
        'subject_id': subject_ids[i],
        'hadm_id': hadm_ids[i],
        'admittime': admit_date,
        'dischtime': discharge_date,
        'deathtime': death_time,
        'admission_type': 'EMERGENCY' if admitted_through_ed else 'ELECTIVE',
        'edregtime': ed_reg_time,
        'edouttime': ed_out_time,
        'los_days': los_days  # Add LOS for convenience
    })

# Create admissions DataFrame
admissions_df = pd.DataFrame(admit_data)

# -------------------- 2. Create diagnoses data -------------------- #

# Initialize list for diagnoses data
diagnoses_data = []

# Track comorbidity scores for verification
comorbidity_scores = {}

for i in range(patient_count):
    # Determine number of diagnoses for this patient (3-10)
    num_diagnoses = np.random.randint(3, 11)
    
    # Randomly select diagnoses categories
    selected_categories = random.sample(list(icd10_categories.keys()), 
                                        min(num_diagnoses, len(icd10_categories)))
    
    # Calculate comorbidity score
    c_score = 0
    comorbidity_details = []
    
    for category in selected_categories:
        # Select a random ICD10 code from the category
        icd_code = random.choice(icd10_categories[category])
        
        # Add to diagnoses data
        diagnoses_data.append({
            'subject_id': subject_ids[i],
            'hadm_id': hadm_ids[i],
            'icd_code': icd_code,
            'icd_version': 10,
            'diagnosis_description': category
        })
        
        # Add to comorbidity score based on the scoring chart
        score_value = 0
        if category in ['Previous myocardial infarction', 'Cerebrovascular disease', 
                        'Peripheral vascular disease', 'Diabetes without complications']:
            score_value = 1
        elif category in ['Congestive heart failure', 'Diabetes with end organ damage',
                          'Chronic pulmonary disease', 'Mild liver disease', 'Any tumor']:
            score_value = 2
        elif category in ['Dementia', 'Connective tissue disease']:
            score_value = 3
        elif category in ['AIDS', 'Moderate or severe liver disease']:
            score_value = 4
        elif category in ['Metastatic solid tumor']:
            score_value = 6
            
        c_score += score_value
        comorbidity_details.append(f"{category} ({score_value})")
    
    # Apply the rule: if total > 3, set C to 5, otherwise C equals total
    final_c_score = 5 if c_score > 3 else c_score
    
    # Store for verification
    comorbidity_scores[subject_ids[i]] = {
        'raw_score': c_score,
        'final_c_score': final_c_score,
        'details': comorbidity_details
    }

# Create diagnoses DataFrame
diagnoses_df = pd.DataFrame(diagnoses_data)

# -------------------- 3. Create prior ED visits data -------------------- #

# Initialize list for prior ED visits
prior_ed_visits_data = []

# Track ED visit counts for verification
ed_visit_counts = {}

for i in range(patient_count):
    # Current admission details
    current_admit_date = admit_data[i]['admittime']
    subject_id = subject_ids[i]
    
    # Determine number of prior ED visits (0-5)
    num_ed_visits = np.random.randint(0, 6)
    
    # Cap at 4 for LACE scoring
    ed_visit_counts[subject_id] = min(num_ed_visits, 4)
    
    # Generate prior ED visits
    for j in range(num_ed_visits):
        # Generate a date within 6 months before current admission
        days_before = np.random.randint(7, 180)
        ed_date = current_admit_date - timedelta(days=days_before)
        
        # ED visits typically last a few hours
        ed_duration_hours = np.random.randint(2, 12)
        
        prior_ed_visits_data.append({
            'subject_id': subject_id,
            'hadm_id': 90000 + len(prior_ed_visits_data),  # Generate unique hadm_id for ED visits
            'edregtime': ed_date,
            'edouttime': ed_date + timedelta(hours=ed_duration_hours),
            'disposition': np.random.choice(['DISCHARGE', 'TRANSFER', 'ADMIT'])
        })

# Create prior ED visits DataFrame
prior_ed_visits_df = pd.DataFrame(prior_ed_visits_data)

# -------------------- 4. Create discharge summaries -------------------- #

discharge_templates = [
    "Patient was admitted for {condition}. Length of stay was {los} days. Patient had a history of {comorbidities}. Patient was {admission_type} and {ed_status}. Patient was discharged in stable condition with follow-up appointments scheduled.",
    "Patient presented with {condition}. During the {los}-day admission, patient was treated for {comorbidities}. This was a {admission_type} admission {ed_status}. Patient was discharged home with outpatient follow-up.",
    "Patient with history of {comorbidities} presented with {condition}. After {los} days of inpatient treatment, patient showed improvement and was discharged. This was a {admission_type} case {ed_status}. Patient advised to follow up with primary care in 1 week.",
    "Patient with {condition} was admitted for {los} days. Medical history includes {comorbidities}. Patient came in through {admission_type} {ed_status}. Discharged in stable condition with medication adjustments."
]

presenting_conditions = [
    "chest pain", "shortness of breath", "abdominal pain", 
    "fever and cough", "dizziness", "altered mental status",
    "lower extremity edema", "acute renal failure", "diabetic ketoacidosis",
    "gastrointestinal bleeding", "syncope", "electrolyte abnormalities"
]

# Create discharge summaries
discharge_summaries = []

for i in range(patient_count):
    subject_id = subject_ids[i]
    hadm_id = hadm_ids[i]
    
    # Get admission details
    los = admit_data[i]['los_days']
    admitted_through_ed = admit_data[i]['admission_type'] == 'EMERGENCY'
    
    # Get comorbidity details
    comorbidity_list = ', '.join(comorbidity_scores[subject_id]['details'])
    if not comorbidity_list:
        comorbidity_list = "no significant chronic conditions"
    
    # Generate discharge summary
    template = random.choice(discharge_templates)
    condition = random.choice(presenting_conditions)
    admission_type = "emergency" if admitted_through_ed else "elective"
    ed_status = "via the Emergency Department" if admitted_through_ed else "as a direct admission"
    
    discharge_text = template.format(
        condition=condition,
        los=los,
        comorbidities=comorbidity_list,
        admission_type=admission_type,
        ed_status=ed_status
    )
    
    discharge_summaries.append({
        'subject_id': subject_id,
        'hadm_id': hadm_id,
        'text': discharge_text
    })

# Create discharge summaries DataFrame
discharge_summaries_df = pd.DataFrame(discharge_summaries)

# -------------------- 5. Calculate LACE scores -------------------- #

# Initialize list for LACE scores
lace_scores = []

for i in range(patient_count):
    subject_id = subject_ids[i]
    hadm_id = hadm_ids[i]
    
    # Calculate L - Length of stay
    los_days = admit_data[i]['los_days']
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
    a_score = 3 if admit_data[i]['admission_type'] == 'EMERGENCY' else 0
    
    # Calculate C - Comorbidity
    c_score = comorbidity_scores[subject_id]['final_c_score']
    
    # Calculate E - ED visits
    e_score = ed_visit_counts.get(subject_id, 0)
    
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

# Create LACE scores DataFrame
lace_scores_df = pd.DataFrame(lace_scores)

# Display the data
print("\n=== PATIENT ADMISSIONS ===")
print(admissions_df[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'admission_type', 'los_days']])

print("\n=== PATIENT DIAGNOSES (Sample) ===")
print(diagnoses_df.head(10))

print("\n=== PRIOR ED VISITS (Sample) ===")
if not prior_ed_visits_df.empty:
    print(prior_ed_visits_df.head(10))
else:
    print("No prior ED visits in the dataset")

print("\n=== DISCHARGE SUMMARIES (Sample) ===")
for i in range(min(3, len(discharge_summaries_df))):
    print(f"\nPatient {discharge_summaries_df.iloc[i]['subject_id']}:")
    print(discharge_summaries_df.iloc[i]['text'])

print("\n=== LACE SCORES ===")
print(lace_scores_df)

# Export as CSV files
admissions_df.to_csv('mimic_admissions.csv', index=False)
diagnoses_df.to_csv('mimic_diagnoses_icd.csv', index=False)
prior_ed_visits_df.to_csv('mimic_ed_visits.csv', index=False)
discharge_summaries_df.to_csv('mimic_discharge_summaries.csv', index=False)
lace_scores_df.to_csv('mimic_lace_scores.csv', index=False)

print("\nCSV files exported successfully.")
