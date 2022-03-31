import torch
from torch import nn
import torch.nn.functional as F
from visualdl.utils.utils import parse_yaml
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import os
import numpy as np

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride = 2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, stride = 2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        return self.double_conv(x)


def predict_series(model, data, device):
    preds = []
    model = model.to(device)
    
    for dt in data:
        tmp = torch.tensor(dt, dtype = torch.float32).permute(1,0).unsqueeze(0)
        tmp = tmp.to(device)
        with torch.cuda.amp.autocast():
            #print(model(tmp).shape)
            predictions, mse = model(tmp)
            predictions = F.softmax(predictions, dim = -1)
            preds.append((predictions[0].detach().cpu().numpy(), mse[0].detach().cpu().numpy()))
    return preds
            

def get_model(classes, num_scalar_outputs,num_fea, use_lstm = True):
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp_cnn = DoubleConv(num_fea,64)
            self.sec_cnn = DoubleConv(64,128)
            self.third = DoubleConv(128,256)
            self.hidden = nn.Linear(256, 1024)
            self.dense1 = nn.Linear(128, 128)
            self.relu = nn.ReLU(inplace = True)
            if classes > 0:
                self.out = nn.Linear(1024, classes)
            if num_scalar_outputs > 0:
                self.mseout = nn.Linear(1024, num_scalar_outputs) 
            self.flatt = nn.Flatten()
            self.lstm = torch.nn.LSTM(input_size = 256, hidden_size = 128, num_layers = 1, batch_first = True, bidirectional = True)
        def forward(self, x):
            x = self.third(self.sec_cnn(self.inp_cnn(x)))
            if use_lstm:
                out,(ht,ct) = self.lstm(x.permute(0,2,1))
                x = torch.cat([ht[0],ht[-1]],dim=1)
            if not use_lstm:
                x = F.avg_pool1d(x, x.size()[2])[:,:,0]
            
            x = self.relu(self.hidden(self.flatt(x)))
            if classes > 0 and num_scalar_outputs == 0:
                return self.out(x)
            elif classes == 0 and num_scalar_outputs > 0:
                return self.mseout(x)
            return self.out(x), self.mseout(x)
    
    return Classifier()

    
        



class PreloadedDataset(Dataset):
    def __init__(self, train_x, train_y):
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y
    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return torch.tensor(self.train_x[idx], dtype = torch.float32).permute(1,0), torch.tensor(self.train_y[idx][0], dtype = torch.long), torch.tensor(self.train_y[idx][1], dtype = torch.float)

class SeriesTrainer():
    def __init__(self, cfg_path:dict):
        self.cfg = parse_yaml(cfg_path)
        assert self.cfg['type'] == "series", "Provided yaml file must be a series config!"
        



    def train(self):
        train_data = None
        valid_data = None
        with open(self.cfg['data']['train'], encoding="utf-8") as handle:
            train_data = json.loads(handle.read())
        with open(self.cfg['data']['valid'], encoding="utf-8") as handle:
            valid_data = json.loads(handle.read())

        # load training data
        train_x = []
        train_y = []
        test_x = []
        test_y = []

        for key, val in train_data.items():
            train_x.append(val['series'])
            train_y.append((val['class'], val['continuous']))

        for key, val in valid_data.items():
            test_x.append(val['series'])
            test_y.append((val['class'], val['continuous']))

        
        train_set = PreloadedDataset(train_x, train_y)
        training_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=self.cfg['settings']['batch_size'], shuffle=True,
                                                    num_workers=0)
        test_set = PreloadedDataset(test_x, test_y)
        valid_loader = torch.utils.data.DataLoader(test_set,
                                                    batch_size=1, shuffle=False,
                                             num_workers=0)
        model = get_model(self.cfg['settings']['outputs']['classes'], self.cfg['settings']['outputs']['continuous'], np.array(train_x[0]).shape[1])
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        epochs = self.cfg['settings']['epochs']
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.cfg['settings']['lr']))
        model.train()
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        mseloss = nn.MSELoss()
        bceloss = nn.BCELoss()
        scaler = torch.cuda.amp.GradScaler()
        best_valid_acc = float("inf")
        for i in range(epochs):
            loss_average = 0.0
            acc_avg = 0.0
            miou_avg = 0.0
            training_bar = tqdm(training_loader)
            model.train()
            for cnt, (x_pred, y_pred, mseunp) in enumerate(training_bar):
                x_pred = x_pred.to(device, dtype=torch.float)
                y_pred = y_pred.to(device, dtype=torch.long)
                mseunp = mseunp.to(device, dtype=torch.float)
                model.zero_grad()
                with torch.cuda.amp.autocast():
                    predictions, mse = model(x_pred)
                    loss = criterion(predictions, y_pred) #+ criterion2(predictions, y_pred)
                    loss += mseloss(mse, mseunp) 
                pred = torch.argmax(predictions, 1)
                acc = (pred == y_pred).float().mean()
                
                loss_average += loss.item()
            
                acc_avg += acc
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                training_bar.set_description("Training: Epoch:%i miou: %.4f, Loss: %.3f,  Acc: %.4f, best: %.2f" % (i,acc_avg / (cnt + 1), float(loss_average / (cnt+1)), acc_avg / (cnt+1), best_valid_acc))
                training_bar.refresh()
            
            #torch.save(model.state_dict(), "glomxDb4f.pt")
            
            with torch.no_grad():    
                test_bar = tqdm(valid_loader)
                loss_average = 0.0
                loss_average_second = 0.0
                model.eval()
                
                acc_avg = 0.0
                miou_avg = 0.0
                for cnt, (x_pred, y_pred, mseunp) in enumerate(test_bar):
                    x_pred = x_pred.to(device, dtype=torch.float)
                    y_pred = y_pred.to(device, dtype=torch.long)
                    mseunp = mseunp.to(device, dtype=torch.float)
                    model.zero_grad()
                    with torch.cuda.amp.autocast():
                        predictions, mse = model(x_pred)
                        loss = criterion(predictions, y_pred) #+ criterion2(predictions, y_pred)
                        loss += mseloss(mse, mseunp)
                    pred = torch.argmax(predictions, 1)
                    loss_average += loss.item()
                    acc = (pred == y_pred).float().mean()
                    acc_avg += acc
                    test_bar.set_description("Valid: Epoch:%i miou: %.4f, Loss: %.3f,Loss: %.3f,  Acc: %.4f, best: %.2f" % (i,acc_avg / (cnt + 1), float(loss_average / (cnt+1)),float(loss_average_second / (cnt+1)), acc_avg / (cnt+1), best_valid_acc))
                if (loss_average / (cnt + 1)) < best_valid_acc:
                    best_valid_acc = loss_average / (cnt + 1)
                    torch.save({
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'classes':self.cfg['settings']['outputs']['classes'],
                    'continous': self.cfg['settings']['outputs']['continuous'],
                    'features': np.array(train_x[0]).shape[1],
                    'custom_data': self.cfg['settings']['custom_data']}, os.path.join(self.cfg['data']['save_folder'], "model.pt"))
                    #torch.save(model.state_dict(), f"bbbb.pt")




        




