# Forked and edited from original authors: https://github.com/haidog-yaqub/Clinical_HybridFusion/tree/main

from models.hybrid_fusion import Hybrid_Fusion
from dataset.diag import Diag
from config import config
from utils.utils import Evaluate
from utils.focal_loss import FocalLoss

import torch
from torch.utils.data import DataLoader
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cls = 28
feature = 31
batch_size = 20
num_works = 8
lr_rate = 1e-5
weight_decay = 1e-5
epochs = 10
report_step = 800

log_val = 'logs/log_val.txt'
log_test = 'logs/log_test.txt'
log_train = 'logs/log_train.txt'
pre_train = None
#  pre_train = 'save/roberta_model.pt'
language_model = "roberta"


if __name__ == "__main__":

    options_name, bert_features, activation_func = config.get_config(language_model)
    bert_features = bert_features[0]

    model = Hybrid_Fusion(
        bert_features=768,  # Explicitly set this to 768
        activation_func=activation_func,
        others_ratio=4,
        input=feature,
        output=cls,
        if_others=False,
        bert_model=options_name  # Pass the model name here
    )
    model = model.to(device)

    data_file = "/Users/shriyareddy/Downloads/discharge_master.csv"

    # loss_func = nn.CrossEntropyLoss()
    loss_func = FocalLoss(None, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)

    # train
    train_data = Diag(
        df=data_file,  # Path to your consolidated dataset
        subset='train',
        label="time_until_next_admission",
        options_name=options_name,
        text='text',  # Using the discharge summary text field
        #others=['los_days', 'prior_ed_visits_count', 'raw_comorbidity_score']  # Only these numeric fields
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_works)

    # val
    
    val_data = Diag(
        df=data_file,  # Path to your consolidated dataset
        subset='val',
        label="time_until_next_admission",
        options_name=options_name,
        text='text',  # Using the discharge summary text field
        #others=['los_days', 'prior_ed_visits_count', 'raw_comorbidity_score']  # Only these numeric fields
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # test
    test_data = Diag(
        df=data_file,  # Path to your consolidated dataset
        subset='test',
        label="time_until_next_admission",
        options_name=options_name,
        text='text',  # Using the discharge summary text field
        #others=['los_days', 'prior_ed_visits_count', 'raw_comorbidity_score']  # Only these numeric fields
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    step = 0
    report_loss = 0.0
    evaluations = []
    accs = []

    if pre_train:
        model.load_state_dict(torch.load(pre_train))
        eval_loss, top_1 = Evaluate(model, val_loader, loss_func, cls, device, 0,
                                    path='val/', language_model=language_model, log=log_val)
        evaluations.append(eval_loss)
        accs.append(top_1)
        print('Val Acc: {:.2f} %'.format(top_1 * 100))

        _, _ = Evaluate(model, test_loader, loss_func, cls,
                        device, 0,
                        path='test/', language_model=language_model, log=log_test)

    model.train()

    for epoch in range(epochs):
        print("\nEpoch is " + str(epoch + 1))

        for i, (text, mask, age, others, label) in enumerate(train_loader):
            optimizer.zero_grad()
            text = text.to(device)
            mask = mask.to(device)

            age = age.to(device)
            others = others.to(device)

            label = label.to(device)

            pred = model(text, mask, age, others)
            loss = loss_func(pred, label)
            report_loss += loss.item()
            loss.backward()
            optimizer.step()

            step += 1
            if (i + 1) % report_step == 0:
                n = open('log/' + language_model + '_' + log_train, mode='a')
                n.write(time.asctime(time.localtime(time.time())))
                n.write('\n')
                n.write('Epoch: [{}][{}]    Batch: [{}][{}]    Loss: {:.6f}\n'.format(
                    epoch + 1, epochs, i + 1, len(train_loader), report_loss / report_step))
                n.close()

                # writer.add_scalar('TrainLoss', report_loss / report_step, step)
                report_loss = 0.0
                # prevent overheating
                time.sleep(10)

        # evaluation
        eval_loss, top_1 = Evaluate(model, val_loader, loss_func, cls,
                                    device, epoch,
                                    path='val/', language_model=language_model, log=log_val)

        evaluations.append(eval_loss)
        accs.append(top_1)
        print('Val Acc: {:.2f} %'.format(top_1 * 100))

        _, _ = Evaluate(model, test_loader, loss_func, cls,
                        device, epoch,
                        path='test/', language_model=language_model, log=log_test)

        # save model
        if len(evaluations) == 1:
            torch.save(model.state_dict(), 'save/'+language_model+'_model.pt')
            n = open('log/'+language_model + '_' + log_val, mode='a')
            n.write('  save=True')
            n.close()
        elif eval_loss <= np.min(evaluations) or top_1 >= np.max(accs):
            torch.save(model.state_dict(), 'save/'+language_model+'_model.pt')
            n = open('log/'+language_model + '_' + log_val, mode='a')
            n.write('  save=True')
            n.close()
        else:
            torch.save(model.state_dict(), 'save/'+language_model+'_model_last.pt')
            n = open('log/'+language_model + '_' + log_val, mode='a')
            n.write('  save=False')
            n.close()