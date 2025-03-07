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
        if bert_model.lower() == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.bert = RobertaForSequenceClassification.from_pretrained('roberta-base')
            self.bert.classifier = nn.Identity()
        elif bert_model.lower() == "clinicalbert":
            self.tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
            self.bert = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        else:
            raise Exception("Unexpected BERT model name:", bert_model, "|| Please select from 'RoBERTa' or 'ClinicalBERT'.")
        
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

    def forward(self, text, mask, others):
        # TODO: FIX FORWARD PASS!!
        x_t = self.bert(text, token_type_ids=None, attention_mask=mask).logits
        if len(x_t.shape) == 3:
            x_t = self.bert_hidden(x_t[:, 0, :])
        else:
            x_t = self.bert_hidden(x_t)

        b, w, h = x_o.shape
        x_o = torch.reshape(x_o, (b, w*h))

        x_o = self.others(x_o)

        x = torch.cat((x_t, x_o), 1)
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