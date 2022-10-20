from matplotlib.style import use
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
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, stride=2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.double_conv(x)


def predict_series(model, data, device, classes, continous, multi_label=False):
    preds = []
    model = model.to(device)
    model.eval()
    for dt in data:
        tmp = torch.tensor(dt, dtype=torch.float32).permute(1, 0).unsqueeze(0)
        tmp = tmp.to(device)
        with torch.cuda.amp.autocast():
            if classes == 0:
                mse = model(tmp)
            elif continous == 0:
                predictions = model(tmp)
            else:
                predictions, mse = model(tmp)
            if classes > 0:
                if not multi_label:
                    predictions = F.softmax(predictions, dim=-1)
                else:
                    predictions = F.sigmoid(predictions, dim=-1)
            if classes > 0 and continous > 0:
                preds.append(
                    (predictions[0].detach().cpu().numpy(), mse[0].detach().cpu().numpy())
                )
            elif classes > 0:
                preds.append((predictions[0].detach().cpu().numpy(), None))
            elif continous > 0:
                preds.append((None,mse[0].detach().cpu().numpy()))
    return preds


def get_model(classes, num_scalar_outputs, num_fea, use_lstm=True):
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp_cnn = DoubleConv(num_fea, 64)
            self.sec_cnn = DoubleConv(64, 128)
            self.third = DoubleConv(128, 256)
            self.hidden = nn.Linear(256, 1024)
            self.dense1 = nn.Linear(128, 128)
            self.relu = nn.ReLU(inplace=True)
            if classes > 0:
                self.out = nn.Linear(1024, classes)
            if num_scalar_outputs > 0:
                self.mseout = nn.Linear(1024, num_scalar_outputs)
            self.flatt = nn.Flatten()
            self.lstm = torch.nn.LSTM(
                input_size=256,
                hidden_size=128,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

        def forward(self, x):
            x = self.third(self.sec_cnn(self.inp_cnn(x)))
            if use_lstm:
                out, (ht, ct) = self.lstm(x.permute(0, 2, 1))
                x = torch.cat([ht[0], ht[-1]], dim=1)
            if not use_lstm:
                x = F.avg_pool1d(x, x.size()[2])[:, :, 0]

            x = self.relu(self.hidden(self.flatt(x)))
            if classes > 0 and num_scalar_outputs == 0:
                return self.out(x)
            elif classes == 0 and num_scalar_outputs > 0:
                return self.mseout(x)
            return self.out(x), self.mseout(x)

    return Classifier()


class PreloadedDataset(Dataset):
    def __init__(self, train_x, train_y, multi_label):
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y
        self.multi_label = multi_label

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        if not self.multi_label:
            return (
                torch.tensor(self.train_x[idx], dtype=torch.float32).permute(1, 0),
                torch.tensor(self.train_y[idx][0], dtype=torch.long),
                torch.tensor(self.train_y[idx][1], dtype=torch.float),
            )
        else:
            return (
                torch.tensor(self.train_x[idx], dtype=torch.float32).permute(1, 0),
                torch.tensor(self.train_y[idx][0], dtype=torch.float),
                torch.tensor(self.train_y[idx][1], dtype=torch.float),
            )


class SeriesTrainer:
    def __init__(self, cfg_path: dict):
        self.cfg = parse_yaml(cfg_path)
        assert (
            self.cfg["type"] == "series"
        ), "Provided yaml file must be a series config!"
        self.multi_label = self.cfg["settings"]["multiple_classes_per_datapoint"]

    def train(self):
        train_data = None
        valid_data = None
        with open(self.cfg["data"]["train"], encoding="utf-8") as handle:
            train_data = json.loads(handle.read())
        with open(self.cfg["data"]["valid"], encoding="utf-8") as handle:
            valid_data = json.loads(handle.read())

        # load training data
        train_x = []
        train_y = []
        test_x = []
        test_y = []

        for key, val in train_data.items():
            train_x.append(val["series"])
            train_y.append((val["class"], val["continuous"]))

        for key, val in valid_data.items():
            test_x.append(val["series"])
            test_y.append((val["class"], val["continuous"]))

        train_set = PreloadedDataset(train_x, train_y, self.multi_label)
        training_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.cfg["settings"]["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        test_set = PreloadedDataset(test_x, test_y, self.multi_label)
        valid_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=False, num_workers=0
        )
        model = get_model(
            self.cfg["settings"]["outputs"]["classes"],
            self.cfg["settings"]["outputs"]["continuous"],
            np.array(train_x[0]).shape[1],
            use_lstm=self.cfg["settings"]["use_lstm"]
        )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        epochs = self.cfg["settings"]["epochs"]
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(self.cfg["settings"]["lr"])
        )
        model.train()
        model.to(device)
        criterion = (
            nn.CrossEntropyLoss() if not self.multi_label else nn.BCEWithLogitsLoss()
        )
        mseloss = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()
        best_valid_acc = float("inf")
        for i in range(epochs):
            loss_average = 0.0
            acc_avg = 0.0
            mse_avg = 0.0
            training_bar = tqdm(training_loader)
            model.train()
            for cnt, (x_pred, y_pred, mseunp) in enumerate(training_bar):
                x_pred = x_pred.to(device)
                y_pred = y_pred.to(device)
                mseunp = mseunp.to(device, dtype=torch.float)
                model.zero_grad()
                with torch.cuda.amp.autocast():
                    out = model(x_pred)
                    if self.cfg['settings']['outputs']['classes'] == 0:
                        mse = model(x_pred)
                        loss = mseloss(mse, mseunp)
                    elif self.cfg['settings']['outputs']['continuous'] == 0:
                        predictions = model(x_pred)
                        loss = criterion(predictions, y_pred)
                    else:
                        predictions, mse = model(x_pred)
                        loss = criterion(predictions, y_pred)
                        loss += mseloss(mse, mseunp)
                if self.cfg['settings']['outputs']['classes'] > 0:    
                    if not self.multi_label:
                        pred = torch.argmax(predictions, 1)
                        acc = (pred == y_pred).float().mean()
                        acc_avg += acc
                    else:
                        pred = predictions > 0.5
                        # acc is binary accuracy for each class
                        acc = (y_pred == pred).sum().item() / y_pred.size(0)
                        acc /= self.cfg["settings"]["outputs"]["classes"]
                        acc_avg += acc
                else:
                    acc = 0
                if self.cfg['settings']['outputs']['continuous'] > 0:
                    mse = torch.abs(mse - mseunp).sum()
                    mse_avg += mse
                    loss_average += loss.item()
                else:
                        mse = 0
                        mse_avg += mse
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                training_bar.set_description(
                    "Training: Epoch:%i, Loss: %.3f,  Acc: %.4f, MSE: %.3f, best: %.2f"
                    % (
                        i,
                        float(loss_average / (cnt + 1)),
                        acc_avg / (cnt + 1),
                        mse_avg / (cnt + 1),
                        best_valid_acc,
                    )
                )
                training_bar.refresh()

            # torch.save(model.state_dict(), "glomxDb4f.pt")

            with torch.no_grad():
                test_bar = tqdm(valid_loader)
                loss_average = 0.0
                model.eval()
                mse_avg = 0.0
                acc_avg = 0.0
                miou_avg = 0.0
                for cnt, (x_pred, y_pred, mseunp) in enumerate(test_bar):
                    x_pred = x_pred.to(device)
                    y_pred = y_pred.to(device)
                    mseunp = mseunp.to(device, dtype=torch.float)
                    model.zero_grad()
                    with torch.cuda.amp.autocast():
                        #predictions, mse = model(x_pred)
                        


                        if self.cfg['settings']['outputs']['classes'] == 0:
                            mse = model(x_pred)
                            loss = mseloss(mse, mseunp)
                        elif self.cfg['settings']['outputs']['continuous'] == 0:
                            predictions = model(x_pred)
                            loss = criterion(predictions, y_pred)
                        else:
                            predictions, mse = model(x_pred)
                            loss = criterion(predictions, y_pred)
                            loss += mseloss(mse, mseunp)


                        #loss = criterion(predictions, y_pred)  
                        #loss += mseloss(mse, mseunp)
                    if self.cfg['settings']['outputs']['classes'] > 0:
                        if not self.multi_label:
                            pred = torch.argmax(predictions, 1)
                            acc = (pred == y_pred).float().mean()
                            acc_avg += acc
                        else:
                            pred = predictions > 0.5
                            # acc is binary accuracy for each class
                            acc = (y_pred == pred).sum().item() / y_pred.size(0)
                            acc /= self.cfg["settings"]["outputs"]["classes"]
                            acc_avg += acc
                    if self.cfg['settings']['outputs']['continuous'] > 0:
                        mse = torch.abs(mse - mseunp).sum()
                        mse_avg += mse
                        loss_average += loss.item()
                    else:
                        mse = 0
                        mse_avg += mse
                    test_bar.set_description(
                        "Valid: Epoch:%i, Loss: %.3f,  Acc: %.4f, MSE: %.3f, best: %.2f"
                        % (
                            i,
                            float(loss_average / (cnt + 1)),
                            acc_avg / (cnt + 1),
                            mse_avg / (cnt + 1),
                            best_valid_acc,
                        )
                    )
                if (loss_average / (cnt + 1)) < best_valid_acc:
                    best_valid_acc = loss_average / (cnt + 1)
                    torch.save(
                        {
                            "epoch": i,
                            "model_state_dict": model.state_dict(),
                            "classes": self.cfg["settings"]["outputs"]["classes"],
                            "continous": self.cfg["settings"]["outputs"]["continuous"],
                            "multi_label": self.multi_label,
                            "features": np.array(train_x[0]).shape[1],
                            "custom_data": self.cfg["settings"]["custom_data"],
                            "use_lstm": self.cfg["settings"]["use_lstm"]
                        },
                        os.path.join(self.cfg["data"]["save_folder"], "model.pt"),
                    )
                    # torch.save(model.state_dict(), f"bbbb.pt")
