# Save this as debug_model.py
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def analyze_model_predictions(model, dataloader, device, num_classes=6):
    """Analyze model predictions to detect if it's always predicting the same class"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for text, mask, age, label in tqdm(dataloader, desc="Analyzing predictions"):
            text = text.to(device)
            mask = mask.to(device)
            age = age.to(device)
            # others = others.to(device)
            
            # outputs = model(text, mask, age, others)
            outputs = model(text, mask, age)

            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Count predictions per class
    pred_counts = np.bincount(all_preds, minlength=num_classes)
    label_counts = np.bincount(all_labels, minlength=num_classes)
    
    # Calculate prediction distribution
    pred_distribution = pred_counts / pred_counts.sum()
    label_distribution = label_counts / label_counts.sum()
    
    # Check if model predictions are heavily biased
    top_5_classes = np.argsort(pred_counts)[-5:][::-1]
    top_5_percentage = pred_counts[top_5_classes].sum() / len(all_preds) * 100
    
    # Compute accuracy
    accuracy = (all_preds == all_labels).mean() * 100
    
    print(f"Model Accuracy: {accuracy:.2f}%")
    print(f"Top 5 predicted classes: {top_5_classes}")
    print(f"Percentage of predictions in top 5 classes: {top_5_percentage:.2f}%")
    
    # Check if any class is predicted more than 10% of the time
    high_freq_classes = np.where(pred_distribution > 0.1)[0]
    if len(high_freq_classes) > 0:
        print(f"WARNING: Classes {high_freq_classes} are predicted more than 10% of the time")
        for cls in high_freq_classes:
            print(f"  Class {cls}: {pred_distribution[cls]*100:.2f}% of predictions but {label_distribution[cls]*100:.2f}% of actual labels")
    
    # Plot prediction distribution
    plt.figure(figsize=(12, 6))
    plt.bar(range(min(30, num_classes)), pred_distribution[:30], label='Predictions')
    plt.bar(range(min(30, num_classes)), label_distribution[:30], alpha=0.5, label='Actual Labels')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of Predictions vs Actual Labels (first 30 classes)')
    plt.savefig('pred_distribution.png')
    
    return all_preds, all_labels, pred_distribution, label_distribution

def check_gradients(model, dataloader, loss_func, device):
    """Check if gradients are flowing through all parts of the model"""
    model.train()
    
    # Get a batch of data
    # text, mask, age, others, label = next(iter(dataloader))
    text, mask, age, others, label = next(iter(dataloader))
    text = text.to(device)
    mask = mask.to(device)
    age = age.to(device)
    # others = others.to(device)
    label = label.to(device)
    
    # Zero gradients
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None
    
    # Forward pass
    # outputs = model(text, mask, age, others)
    outputs = model(text, mask, age)
    loss = loss_func(outputs, label)
    
    # Backward pass
    loss.backward()
    
    # Check gradients for different components
    components = {
        'bert_hidden': model.bert_hidden,
        'age_embedding': model.age_embedding,
        'output_layer': model.output
    }
    
    for name, component in components.items():
        has_grad = False
        no_grad_params = 0
        total_params = 0
        
        for param in component.parameters():
            total_params += 1
            if param.requires_grad:
                if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                    has_grad = True
                else:
                    no_grad_params += 1
        
        if has_grad:
            print(f"{name}: Gradients flowing! ({total_params-no_grad_params}/{total_params} parameters)")
        else:
            print(f"{name}: NO GRADIENTS DETECTED! ({no_grad_params}/{total_params} parameters without gradients)")
    
    # Check if BERT is frozen (it should be in your current setup)
    bert_requires_grad = [p.requires_grad for p in model.bert.parameters()]
    if any(bert_requires_grad):
        print("WARNING: Some BERT parameters have requires_grad=True!")
    else:
        print("BERT is frozen as expected.")

# Updated Enhanced Hybrid_Fusion model
class EnhancedHybridFusion(torch.nn.Module):
    """Enhanced version of Hybrid_Fusion with fixes for the accuracy issue"""
    def __init__(self, original_model):
        super(EnhancedHybridFusion, self).__init__()
        
        # Copy all attributes from the original model
        self.bert = original_model.bert
        self.tokenizer = original_model.tokenizer
        self.bert_hidden = original_model.bert_hidden
        self.age_embedding = original_model.age_embedding
        self.binary_embedding = original_model.binary_embedding
        self.others = original_model.others
        self.output = original_model.output
        self.if_others = original_model.if_others
        
        # Add an additional layer to break symmetry
        if hasattr(original_model.output, "layers"):
            # If output is a ModuleList, get the output size of the last layer
            last_layer = original_model.output[-1]
            if isinstance(last_layer, torch.nn.Linear):
                output_size = last_layer.out_features
        else:
            # If output is a Sequential, find the last Linear layer
            for module in reversed(list(original_model.output.modules())):
                if isinstance(module, torch.nn.Linear):
                    output_size = module.out_features
                    break
        
        # Add a small additional layer with a slightly different initialization
        self.final_layer = torch.nn.Linear(output_size, output_size)
        # Initialize with slightly random weights to break symmetry
        torch.nn.init.xavier_uniform_(self.final_layer.weight, gain=0.01)
        
        # By default, unfreeze BERT for fine-tuning
        self.unfreeze_bert()
    
    def forward(self, input_ids, attention_mask, age, others=None):
        # Process text through BERT
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
        
        # Apply the final layer with different initialization to break symmetry
        x = self.final_layer(x)
        
        # Apply softmax to ensure proper probability distribution
        # x = torch.nn.functional.softmax(x, dim=1)
        
        return x
    
    def freeze_bert(self):
        self.bert.eval()  # Set to evaluation mode
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert(self):
        self.bert.train()  # Set to training mode
        for param in self.bert.parameters():
            param.requires_grad = True
    
    def enable_effective_grad_checkpointing(self):
        """Enable gradient checkpointing properly"""
        if hasattr(self.bert, 'gradient_checkpointing_enable'):
            self.bert.gradient_checkpointing_enable()
        
        # For other components, manually implement checkpointing if needed
        # Note: Sequential modules don't natively support gradient checkpointing

# Example usage in train.py:

"""
# First analyze your current model
from debug_model import analyze_model_predictions, check_gradients, EnhancedHybridFusion

# Check if model is predicting the same classes for all inputs
preds, labels, pred_dist, label_dist = analyze_model_predictions(model, val_loader, device)

# Check if gradients are flowing through all parts
check_gradients(model, train_loader, loss_func, device)

# If issues are detected, wrap your model with the enhanced version
enhanced_model = EnhancedHybridFusion(model)
enhanced_model = enhanced_model.to(device)

# Use a proper learning rate scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
optimizer = torch.optim.Adam(
    [
        {"params": enhanced_model.bert.parameters(), "lr": 1e-5},  # Lower LR for BERT
        {"params": list(enhanced_model.age_embedding.parameters()) + 
                  list(enhanced_model.binary_embedding.parameters()) +
                  list(enhanced_model.bert_hidden.parameters()) +
                  list(enhanced_model.others.parameters()) +
                  list(enhanced_model.output.parameters()) +
                  list(enhanced_model.final_layer.parameters()), "lr": 5e-4}
    ],
    weight_decay=1e-5
)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

# Then train the enhanced model...
"""