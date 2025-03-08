import torch
from torch import nn

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig


class Hybrid_Fusion(nn.Module):

    def __init__(self, bert_features=768, activation_func=nn.Tanh(),
                 hidden=384, others_ratio=4, input=0, output=24, if_others=False,
                 bert_model : str = "roberta"):

        super(Hybrid_Fusion, self).__init__()
        self.if_others = if_others

        # Import BERT and strip classification layer
        if bert_model.lower() == "roberta-base":
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.bert = RobertaForSequenceClassification.from_pretrained('roberta-base')
            self.bert.classifier = nn.Identity()
        elif bert_model.lower() == "clinicalbert":
            self.tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
            self.bert = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        elif bert_model.lower() == "clinical-longformer" or bert_model == "yikuan8/Clinical-Longformer":
            self.tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
            self.bert = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
        else:
            raise Exception("Unexpected BERT model name:", bert_model, "|| Please select from 'RoBERTa' or 'ClinicalBERT'.")
        
        self.age_embedding = nn.Linear(1, hidden//others_ratio)
        self.binary_embedding = nn.Embedding(2, hidden//others_ratio)
        
        # Post-BERT hidden layer for training embeddings
        self.bert_hidden = nn.Sequential(nn.Dropout(0.1),
                                         nn.Linear(bert_features, hidden, bias=True),
                                         activation_func,
                                         nn.Dropout(0.1),
                                         )

        # TODO: Plug in XGBoost Model
        self.others = nn.Sequential(nn.Linear((input+1)*3, hidden//others_ratio),
                                    nn.LayerNorm(hidden//others_ratio, eps=1e-12),
                                    activation_func,
                                    nn.Dropout(0.1),
                                    nn.Linear(hidden//others_ratio, hidden//others_ratio),
                                    nn.LayerNorm(hidden//others_ratio, eps=1e-12),
                                    activation_func,
                                    nn.Dropout(0.1),
                                    )

        self.output = nn.Sequential(nn.Linear(hidden+hidden//others_ratio, hidden),
                                    nn.LayerNorm(hidden, eps=1e-12),
                                    activation_func,
                                    nn.Dropout(0.1),
                                    nn.Linear(hidden, hidden),
                                    nn.LayerNorm(hidden, eps=1e-12),
                                    activation_func,
                                    nn.Dropout(0.1),
                                    nn.Linear(hidden, output)
        )

    def forward(self, input_ids, attention_mask, age, others=None):
        """
        Forward pass for the Hybrid_Fusion model.
        
        Args:
            input_ids (torch.Tensor): Token ids for the text input to BERT
            attention_mask (torch.Tensor): Attention mask for BERT
            age (torch.Tensor): Age values of shape [batch_size, 1]
            others (torch.Tensor, optional): Other features if if_others is True
            
        Returns:
            torch.Tensor: Model outputs
        """
        # Process text through BERT (maintaining compatibility with all model types)
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle different output formats from various BERT models
        if hasattr(bert_outputs, 'logits'):
            x_t = bert_outputs.logits
        else:
            # For models that return last_hidden_state
            x_t = bert_outputs.last_hidden_state[:, 0, :]

        if len(x_t.shape) == 3:
            x_t = x_t[:, 0, :]  # Take the first token ([CLS]) representation BEFORE bert_hidden

        # Pass through BERT hidden layer
        x_t = self.bert_hidden(x_t)
        
        # Process age data
        x_age = self.age_embedding(age)
        batch_size, hidden_dim = x_age.shape
        x_age = x_age.view(batch_size, 1, hidden_dim)  # Reshape to 3D: [batch, 1, hidden//others_ratio]
        
        # Process other features if applicable
        if self.if_others and others is not None:
            x_others = self.binary_embedding(others.long())
            x_o = torch.cat((x_age, x_others), dim=1)  # Concatenate along sequence dimension
        else:
            x_o = x_age
        
        # Flatten the 3D tensor to 2D for the others network
        batch_size, seq_len, hidden_dim = x_o.shape
        x_o = x_o.reshape(batch_size, seq_len * hidden_dim)
        
        # Process through the others network
        x_o = self.others(x_o)
        
        # Concatenate BERT and others embeddings
        x = torch.cat((x_t, x_o), dim=1)
        
        # Pass through output layer
        x = self.output(x)
        
        return x


# options_name = "yikuan8/Clinical-Longformer"
# tokenizer = AutoTokenizer.from_pretrained(options_name, model_max_length=256)
# text = 'lets go'
#
# inputs = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")
# text = inputs['input_ids']
# mask = inputs['attention_mask']
#
# age = torch.rand(1, 1)
# others = torch.tensor([[1, 0]])
#
# M = BERT_multi(options_name, 768, nn.Tanh(), 384, 0, 8)
#
# y = M(text, mask, age, others)