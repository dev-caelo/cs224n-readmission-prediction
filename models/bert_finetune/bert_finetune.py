import torch.nn as nn
from transformers import AutoModel

class ClinicalBERTClassifier(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.1):
        """
        Initialize the ClinicalBERTClassifier with ClinicalBERT as the base model for finetuning.
        
        num_classes (int): Number of classification classes
        dropout_rate (float): Dropout rate for the classification head
        """
        super(ClinicalBERTClassifier, self).__init__()
        
        # Load pre-trained ClinicalBERT model
        self.bert = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        
        # Get the hidden size from the BERT configuration
        hidden_size = self.bert.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Token IDs from the tokenizer
            attention_mask (torch.Tensor): Attention mask from the tokenizer
            
        Returns:
            torch.Tensor: Logits for each class
        """
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through the classification head
        logits = self.classifier(cls_output)
        
        return logits