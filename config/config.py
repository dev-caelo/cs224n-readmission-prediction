# Forked and edited from original authors: https://github.com/haidog-yaqub/Clinical_HybridFusion/blob/main/config/get_config.py

from torch.nn import Tanh

def get_config(language_model):
    if language_model == 'bert':
        model_name = "bert-base-uncased"
        bert_features = 768
        activation_func = Tanh()
    elif language_model == "clinical_longformer":
        model_name = "yikuan8/Clinical-Longformer"
        bert_features = 768
        activation_func = Tanh()
    elif language_model == "roberta":
        model_name = "roberta-base"
        bert_features = 768
        activation_func = Tanh()
    elif language_model == "clinical_bert":
        model_name = "clinicalbert"
        bert_features = 768
        activation_func = Tanh()
    else:
        print('supported models: bert, clinical_longformer, roberta')
        return 'error'

    return model_name, bert_features, activation_func