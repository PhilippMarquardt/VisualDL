import torch
from torch import nn
from visualdl.utils.utils import parse_yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import json
import numpy as np
import pandas as pd
from openpyxl import load_workbook


def predict_mlp(model, data, device):
    preds = []
    model = model.to(device)

    for dt in data:
        tmp = torch.tensor(dt)
        tmp = tmp.to(device)
        with torch.cuda.amp.autocast():
            prediction = model(tmp)
            preds.append(prediction.detach().cpu().numpy())
    return preds


def get_mlp_model(in_features: int, out_features: int):
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(in_features, 100)
            self.relu1 = torch.nn.ReLU()           
            self.fc2 = torch.nn.Linear(100, 500)
            self.relu2 = torch.nn.ReLU()
            self.fc3 = torch.nn.Linear(500, out_features)
           
            
        def forward(self, x):
            hidden1 = self.fc1(x)
            relu1 = self.relu1(hidden1)
            hidden2 = self.fc2(relu1)
            relu2 = self.relu1(hidden2)
            output = self.fc3(relu2)
            return output
    
    return MLP()


class SpectraPreloadedDataset(Dataset):
    """Dataset of spectra prediction which uses preloaded data"""
    def __init__(self, data_path: str):
        super().__init__()

        data = None
        with open(data_path, encoding="utf-8") as handle:
            data = json.loads(handle.read())

        self.data_x, self.data_y = [], []
        for key, val in data.items():
            self.data_x.append(val['concentrations'])
            self.data_y.append(val['spectrum'])


    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return torch.tensor(self.data_x[idx]), torch.tensor(self.data_y[idx])



class MLPDataset(Dataset):
    """Template for general MLP dataset."""
    def __init__(self, csv_path: str):
        super().__init__()

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return self.train_x[idx], self.train_y[idx]



class MLPTrainer():
    def __init__(self, cfg_path: str):
        self.cfg = parse_yaml(cfg_path)
        assert self.cfg['type'] == "mlp", "Provided yaml file must be a regression config!"


    def train(self):
        # get training data
        train_set = SpectraPreloadedDataset(self.cfg['data']['train'])
        training_loader = DataLoader(train_set, batch_size=self.cfg['settings']['batch_size'], shuffle=True, num_workers=0)

        validation_set = SpectraPreloadedDataset(self.cfg['data']['valid'])
        validation_loader = DataLoader(validation_set, batch_size=self.cfg['settings']['batch_size'], shuffle=True, num_workers=0)

        # get model and set hyperparametters
        model = get_mlp_model(self.cfg['settings']['in_features'], self.cfg['settings']['out_features'])
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        epochs = self.cfg['settings']['epochs']
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.cfg['settings']['lr']))
        criterion = nn.MSELoss()

        model.train()
        model.to(device)

        scaler = torch.cuda.amp.GradScaler()
        
        best_valid_loss = float("inf")
        early_stopping_patience = self.cfg['settings']['early_stopping_patience']
        early_stopping_counter = 0
        for i in range(epochs):
            loss_average = 0.0
            training_bar = tqdm(training_loader)
            model.train()
            for cnt, (x_pred, y_pred) in enumerate(training_bar): # what ist output of dataloader?
                x_pred = x_pred.to(device)
                y_pred = y_pred.to(device)
                model.zero_grad()
                with torch.cuda.amp.autocast():
                    predictions = model(x_pred)
                    loss = criterion(predictions, y_pred[:,:,1])
                
                loss_average += loss.item()
            
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                training_bar.set_description("Training: Epoch:%i, Loss: %.3f, best: %.2f" % (i, float(loss_average / (cnt+1)), best_valid_loss))
                training_bar.refresh()
        
            # validation
            with torch.no_grad():    
                test_bar = tqdm(validation_loader)
                loss_average = 0.0
                model.eval()
            
                for cnt, (x_pred, y_pred) in enumerate(test_bar):
                    x_pred = x_pred.to(device)
                    y_pred = y_pred.to(device)
                    model.zero_grad()
                    with torch.cuda.amp.autocast():
                        predictions = model(x_pred)
                        loss = criterion(predictions, y_pred[:,:,1])
                    
                    loss_average += loss.item()
                    test_bar.set_description("Valid: Epoch:%i, Loss: %.3f, best: %.2f" % (i, float(loss_average / (cnt+1)), best_valid_loss))

                if (loss_average / (cnt + 1)) >= best_valid_loss:
                    early_stopping_counter += 1
                    if early_stopping_counter == early_stopping_patience:
                        print("Early stopping")
                        break

                if (loss_average / (cnt + 1)) < best_valid_loss:
                    best_valid_loss = loss_average / (cnt + 1)
                    early_stopping_counter = 0
                    torch.save({
                        'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'in_features': self.cfg['settings']['in_features'],
                        'out_features': self.cfg['settings']['out_features'],
                        'custom_data': self.cfg['settings']['custom_data']}, os.path.join(self.cfg['data']['save_folder'], "model.pt"))
