# Modified train.py with frozen BERT for faster training

from models.hybrid_fusion import Hybrid_Fusion
from dataset.torch_diag import DiagTorch
from config import config
from utils.utils import Evaluate
from utils.focal_loss import FocalLoss
from tqdm import tqdm
from debug_model import analyze_model_predictions, check_gradients, EnhancedHybridFusion
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import os
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



cls = 2    # THE NUMBER OF BINS
feature = 32
batch_size = 20  # Keeping original batch size
num_works = 8
lr_rate = 8e-4  # Adjusted learning rate for frozen BERT setup
weight_decay = 1e-4
epochs = 5  # Increased epochs
report_step = 500  # Report more frequently

log_val = 'log_val.txt'
log_test = 'log_test.txt'
log_train = 'log_train.txt'
pre_train = None

language_model = None

# Create necessary directories
directories = ['log', 'logs', 'save', 'visualizations']
for directory in directories:
    os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='modified_train')
    parser.add_argument('model', help="BERT model to run: 'RoBERTa' || 'Clinical_BERT'")
    parser.add_argument('bins', help="Either 2 (LACE) or 6")

    # Update params using args
    args = parser.parse_args()
    language_model = str(args.model).lower()
    options_name, bert_features, activation_func = config.get_config(language_model=language_model)
    cls = int(args.bins)

    # Initialize the base model
    model = Hybrid_Fusion(
        bert_features=768,
        activation_func=activation_func,
        others_ratio=4,
        input=feature,
        output=cls,
        if_others=False,
        bert_model=options_name
    )
    model = model.to(device)
    
    # Enhance the model to fix potential issues
    model = EnhancedHybridFusion(model)
    model = model.to(device)

    # Define which columns to include from data
    columns_to_include = ['text', 'anchor_age', 'gender', 'race', 'hadm_id',
                          'admission_location', 'discharge_location', 'insurance', 'admission_type',
                          'note_type', 'language', 'marital_status', 'diagnoses_long_title',
                          'diagnoses_icd', 'procedures_long_title', 'procedures_icd', 'note_seq']
    
    # Explicitly freeze BERT for faster training
    model.freeze_bert()

    data_file = "dataset/new_discharge_master.csv"

    # Use weighted loss function for better handling of class imbalance
    class_weights = torch.ones(cls, device=device)

    """
    MANUALLY ADJUST BIN WEIGHTS HERE
    - This is a quick fix to deal with our data imbalance.
    - Remember that this is [inclusive, exclusive) -> [0:10] = (0, 1, 2, ..., 9)
    """

    """"""

    loss_func = nn.CrossEntropyLoss(weight=class_weights)
    
    # Since BERT is frozen, optimize only the rest of the network
    optimizer = torch.optim.Adam([
        {"params": list(model.age_embedding.parameters()) + 
                  list(model.binary_embedding.parameters()) +
                  list(model.bert_hidden.parameters()) +
                  list(model.others.parameters()) +
                  list(model.output.parameters()) +
                  list(model.final_layer.parameters()), "lr": lr_rate}
    ], weight_decay=weight_decay)
    
    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, verbose=True)

    # Load training data
    train_data = DiagTorch(
        data_path=data_file,
        subset='train',
        label="time_until_next_admission",
        tokenizer=model.tokenizer,
        #text='text',
        total_bins=cls,
        columns_to_include=columns_to_include
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_works)

    # Load validation data
    val_data = DiagTorch(
        data_path=data_file,
        subset='val',
        label="time_until_next_admission",
        tokenizer=model.tokenizer,
        #text='text',
        columns_to_include=columns_to_include
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Load test data
    test_data = DiagTorch(
        data_path=data_file,
        subset='test',
        label="time_until_next_admission",
        tokenizer=model.tokenizer,
        #text='text',
        columns_to_include=columns_to_include
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Check model output distributions before training
    # print("Analyzing model before training...")
    # preds, labels, pred_dist, label_dist = analyze_model_predictions(model, val_loader, device)
    
    # Check if gradients are flowing properly (except for BERT which is frozen)
    # print("\nChecking gradients before training...")
    # check_gradients(model, train_loader, loss_func, device)

    step = 0
    report_loss = 0.0
    evaluations = []
    accs = []

    if pre_train:
        model.load_state_dict(torch.load(pre_train))
        eval_loss, top_1 = Evaluate(model, val_loader, loss_func, cls, device, 0,
                                    path='log/', language_model=language_model, log=log_val)
        evaluations.append(eval_loss)
        accs.append(top_1)
        print(f'Val Acc: {top_1 * 100:.2f} %')

    model.train()
    # Keep BERT in eval mode since it's frozen
    model.bert.eval()
    
    # Use torch.no_grad for BERT to speed up forward pass
    def forward_with_frozen_bert(model, input_ids, attention_mask, age, others=None):
        with torch.no_grad():
            bert_outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle different output formats from various BERT models
            if hasattr(bert_outputs, 'logits'):
                x_t = bert_outputs.logits
            else:
                # For models that return last_hidden_state
                x_t = bert_outputs.last_hidden_state[:, 0, :]

            if len(x_t.shape) == 3:
                x_t = x_t[:, 0, :]  # Take the first token ([CLS]) representation
                
            # Clone the tensor to detach it from the computation graph
            x_t = x_t.clone()
            
        # Outside no_grad context, we can now set requires_grad
        x_t.requires_grad_(True)
            
        # Continue with the rest of the forward pass
        x_t = model.bert_hidden(x_t)
        
        # Process age data
        x_age = model.age_embedding(age)
        batch_size, hidden_dim = x_age.shape
        x_age = x_age.view(batch_size, 1, hidden_dim)
        
        # Process other features if applicable
        if model.if_others and others is not None:
            x_others = model.binary_embedding(others.long())
            x_o = torch.cat((x_age, x_others), dim=1)
        else:
            x_o = x_age
        
        # Flatten the 3D tensor to 2D for the others network
        batch_size, seq_len, hidden_dim = x_o.shape
        x_o = x_o.reshape(batch_size, seq_len * hidden_dim)
        
        # Process through the others network
        x_o = model.others(x_o)
        
        # Concatenate BERT and others embeddings
        x = torch.cat((x_t, x_o), dim=1)
        
        # Pass through output layer
        x = model.output(x)
        
        # Apply the final layer with different initialization
        x = model.final_layer(x)
        
        return x

    for epoch in range(epochs):
        print(f"\nEpoch is {epoch + 1}")
        
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for i, (text, mask, age, label) in enumerate(progress_bar):
            start = time.time()
            optimizer.zero_grad()
            
            text = text.to(device)
            mask = mask.to(device)
            age = age.to(device)
            #others = others.to(device)
            label = label.to(device)

            # Use the custom forward function with frozen BERT
            pred = forward_with_frozen_bert(model, text, mask, age)
            
            # Check for NaN outputs (debug)
            if torch.isnan(pred).any():
                print("\nWARNING: NaN detected in model output!")
                continue
                
            loss = loss_func(pred, label)
            
            # Gradient clipping to prevent exploding gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            report_loss += current_loss
            num_batches += 1
            
            step += 1
            end = time.time()
            
            # For the first few batches, check prediction distribution
            # if epoch == 0 and i < 3:
            #     _, predicted = torch.max(pred, 1)
            #     pred_classes = predicted.cpu().numpy()
            #     unique_classes, counts = np.unique(pred_classes, return_counts=True)
            #     print(f"\nBatch {i} predictions: {list(zip(unique_classes, counts))}")
            
            progress_bar.set_postfix({"Loss": current_loss, "Time": end-start})
            
            if (i + 1) % report_step == 0:
                # Calculate accuracy for this batch
                _, predicted = torch.max(pred, 1)
                batch_acc = (predicted == label).float().mean().item() * 100
                
                log_msg = f'Epoch: [{epoch + 1}/{epochs}] Batch: [{i + 1}/{len(train_loader)}] ' \
                         f'Loss: {report_loss / report_step:.6f} Accuracy: {batch_acc:.2f}%\n'
                
                with open('log/' + language_model + '_' + log_train, mode='a') as n:
                    n.write(time.asctime(time.localtime(time.time())) + '\n')
                    n.write(log_msg)
                
                print(f"\n{log_msg}")
                report_loss = 0.0
        
        # Epoch complete - calculate average loss
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.6f}")
        
        # Evaluate on validation set
        model.eval()  # Set model to evaluation mode
        eval_loss, top_1 = Evaluate(model, val_loader, loss_func, cls, device, epoch,
                                    path='log/', language_model=language_model, log=log_val)
        model.train()  # Set back to training mode
        model.bert.eval()  # Keep BERT in eval mode
        
        evaluations.append(eval_loss)
        accs.append(top_1)
        print(f'Val Acc: {top_1 * 100:.2f} %')
        
        # Step the scheduler
        scheduler.step(eval_loss)
        
        # Check if we've made significant progress
        if len(accs) > 1 and abs(accs[-1] - accs[-2]) < 0.001:
            print("\nWarning: Training has stagnated! Trying with a 10x smaller learning rate...")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                
        # After every 3 epochs, analyze model predictions
        if (epoch + 1) % 3 == 0:
            print(f"\nAnalyzing model predictions after epoch {epoch+1}...")
            analyze_model_predictions(model, val_loader, device, num_classes=cls)

        # Save model
        if len(evaluations) == 1:
            torch.save(model.state_dict(), 'save/'+language_model+'_model.pt')
            with open('log/'+language_model + '_' + log_val, mode='a') as n:
                n.write('  save=True')
        elif eval_loss <= np.min(evaluations) or top_1 >= np.max(accs):
            torch.save(model.state_dict(), 'save/'+language_model+'_model.pt')
            with open('log/'+language_model + '_' + log_val, mode='a') as n:
                n.write('  save=True')
        else:
            torch.save(model.state_dict(), 'save/'+language_model+'_model_last.pt')
            with open('log/'+language_model + '_' + log_val, mode='a') as n:
                n.write('  save=False')

    # Final test evaluation
    _, top_1 = Evaluate(model, test_loader, loss_func, cls, device, epochs-1,
                    path='log/', language_model=language_model, log=log_test)
    print(f'FINAL -> Test Acc: {top_1 * 100:.2f} %')