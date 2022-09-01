import torch
from torch import nn
from visualdl.utils.utils import parse_yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from openpyxl import load_workbook


def predict_mlp(model, data, device):
    preds = []
    model = model.to(device)

    for dt in data:
        tmp = torch.tensor(dt)
        tmp = tmp.to(device)
        with torch.cuda.amp.autocast(): # ???
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


class MLPDataset(Dataset):
    def __init__(self, csv_path: str):
        super().__init__()
        self.data_path = csv_path

        # get input and output for model training from csv files
        self.train_x, self.train_y = [], []

        # gemische
        file_path = r"C:\Users\HSA\Desktop\Spektren\Gemische\Pulvermischungen Tabletten_neue Einwaage für M44_1.xlsx"
        # spectra_dir = r"C:\Users\HSA\Desktop\Spektren\Gemische\csv Dateien\train"
        spectra_dir = csv_path
        wb = load_workbook(filename = file_path, data_only=True)
        sheet = wb['Zusammenfassung']

        spectra_files = os.listdir(spectra_dir)
        for i in range(3,67):
            # print(sheet[f'A{i}'].value)
            file_name = sheet[f'A{i}'].value
            file_number = file_name[-2:]
            file_number = file_number if file_number[0] != "0" else file_number[1]
            samples = [x for x in spectra_files if ("Tablette " + file_number + "_") in x]
            # print(samples)
            
            for sample in samples:        
                with open(file_path, encoding = "utf-8") as handle:
                    csv_data = pd.read_csv(os.path.join(spectra_dir, sample), sep = ';',skiprows=list(range(0, 89)), header=None)
            
                all_values = np.array(list(map(lambda x: [float(x[i].replace(",", ".")) if type(x[i]) is str else x[i] for i in range(2)], csv_data.values.tolist())), dtype=np.float32)
                idx = np.round(np.linspace(0, len(all_values) - 1, 100)).astype(int)
                self.train_y.append(list(all_values[idx]))
                
                self.train_x.append([sheet[f'G{i}'].value, sheet[f'F{i}'].value, sheet[f'H{i}'].value])



        # einzelne substanzen
        # for cnt, csv in enumerate(os.listdir(csv_path)):
        #     csv_file_path = os.path.join(csv_path, csv)
        #     csv_data = self.get_csv_data(csv_file_path)
            
        #     if ("Chromotrope") in csv:
        #         if csv.split(",")[0][-1] == "2":
        #             val = 12.
        #         elif  csv.split(",")[0][-1] == "0":
        #             val = 0.5
        #         elif  csv.split(",")[0][-1] == "1":
        #             val = 1.5
        #         else:
        #             val =  csv.split(",")[0][-1]
        #         val = float(val)
        #         self.train_x.append([val, 0, 0])
        #     elif ("Chromtrope") in csv:
        #         if csv.split(",")[0][-1] == "2":
        #             val = 12.
        #         elif  csv.split(",")[0][-1] == "0":
        #             val = 0.5
        #         elif  csv.split(",")[0][-1] == "1":
        #             val = 1.5
        #         else:
        #             val =  csv.split(",")[0][-1]
        #         val = float(val)
        #         self.train_x.append([val, 0, 0])
        #     elif "Erioglaucine" in csv:
        #         if csv.split(".")[0][-1] == "2":
        #             val = 12.
        #         elif  csv.split(".")[0][-1] == "0":
        #             val = 0.5
        #         elif  csv.split(".")[0][-1] == "1":
        #             val = 1.5
        #         else:
        #             val =  csv.split(".")[0][-1]
        #         val = float(val)
        #         self.train_x.append([0, val, 0])
        #     elif "Riboflavin" in csv:
        #         if csv.split(",")[0][-1] == "2":
        #             val = 12.
        #         elif  csv.split(",")[0][-1] == "0":
        #             val = 0.5
        #         elif  csv.split(",")[0][-1] == "1":
        #             val = 1.5
        #         else:
        #             val =  csv.split(",")[0][-1]
        #         val = float(val)
        #         self.train_x.append([0, 0, val])
        #     else:
        #         print(f"no valid substance: {csv}")
        #         continue
            
        #     all_values = np.array(list(map(lambda x: [float(x[i].replace(",", ".")) if type(x[i]) is str else x[i] for i in range(2)], csv_data.values.tolist())), dtype=np.float32)
        #     idx = np.round(np.linspace(0, len(all_values) - 1, 100)).astype(int)
        #     self.train_y.append(list(all_values[idx]))


    def get_csv_data(self, file_path):
        with open(file_path, encoding = "utf-8") as handle:
            data = pd.read_csv(file_path, sep = ';',skiprows=list(range(0, 89)), header=None)
        return data

    def __len__(self):
        # return number of csv files
        return len(os.listdir(self.data_path))

    def __getitem__(self, idx):
        return torch.tensor(self.train_x[idx]), torch.tensor(self.train_y[idx])



class MLPTrainer():
    def __init__(self, cfg_path:dict):
        self.cfg = parse_yaml(cfg_path)
        assert self.cfg['type'] == "mlp", "Provided yaml file must be a regression config!"


    def train(self):
        # get training data
        train_set = MLPDataset(self.cfg['data']['train'])
        training_loader = DataLoader(train_set, batch_size=self.cfg['settings']['batch_size'], shuffle=True, num_workers=0)

        validation_set = MLPDataset(self.cfg['data']['valid'])
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
        
