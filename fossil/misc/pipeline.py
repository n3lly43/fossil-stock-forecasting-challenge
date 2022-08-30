import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import re
import time
import random
import numpy as np
from time import time
from pathlib import Path
from ..config import ModelsConfig
        
class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val

class ModelTrainer:
    def train_model(self, early_stop=3, verbose=10, save_path=None):
        
        stop_eps = 0 
        best_epoch = None
        metrics = {'MAE':[]}
        start_time = time()
        accum_mae = torch.tensor(float('inf')) if not self.resume_train else torch.tensor(float(re.findall("\d+\.\d+", self.model_weights)[0]))
        try:
            for epoch in range(1, ModelsConfig.EPOCHS + 1):
                train_loss = self.train_step(epoch, 1500)             
                test_loss, w8, mae = self.val_step(epoch)

                if mae < accum_mae:
                    best_epoch = epoch
                    accum_mae = mae
                    val_loss = test_loss

                    self.model_checkpoints.append((w8, test_loss, mae))
                    stop_eps = 0
                else:
                    stop_eps += 1
                if epoch % verbose ==0:
                    print('Epoch {}: Train Loss: {:.4f}\tVal Loss: {:.4f}\tVal MAE: {:.2f}\telapsed: {:.2f} mins'.format(
                    epoch, torch.mean(train_loss), test_loss, mae, (time()-start_time)/60))
                

                if stop_eps >= early_stop:
                    self.save_model_checkpoint(save_path)
                    print('\n')
                    print('Early stopping: Best Epoch: {}\tVal loss: {:.4f}\tVal MAE: {:.2f}\tTotal time elapsed: {:.2f} mins'.format(
                        best_epoch, val_loss, accum_mae, (time()-start_time)/60))
                    print('-'*50)
                    print('\n')
                    break

        except KeyboardInterrupt:
            self.save_model_checkpoint(save_path) if self.model_checkpoints else print("first epoch not completed, model checkpoint will not be saved")
            
            if stop_eps<early_stop:
                print(f'Training stopped, early stop not reached. Best Iteration: {best_epoch} Val loss: {accum_loss} Val MAE: {mae}')

    def save_model_checkpoint(self, save_path):
        best_model, accum_loss, metrics = self.model_checkpoints[-1]

        if save_path is not None:
            torch.save(best_model, Path(save_path).joinpath(f'loss_{accum_loss}_MAE_{metrics}_fossil.pth'))

    def load_model_checkpoint(self, save_path):
        self.model.load_state_dict(torch.load(Path(save_path)))            

class FossilPipeline(ModelTrainer):
    def __init__(self, train_dataloader, test_dataloader, model, target_scaler, resume_from_checkpoint=False, model_weights=None):
        
        self.model = model.to(ModelsConfig.device) 
        self.resume_train = resume_from_checkpoint
        self.model_weights = model_weights
        self.target_scaler = target_scaler

        if self.resume_train:
            assert model_weights is not None, "specify model weights save location"
            self.load_model_checkpoint(self.model_weights)

        self.train_loader = train_dataloader
        self.test_loader = test_dataloader

        self.criterion = nn.L1Loss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=ModelsConfig.learning_rate)
        
        self.model_checkpoints = []

    def train_step(self, epoch, verbose):
        self.model.train()
        
        train_loss=[]
        start_time = time()
        data_len = len(self.train_loader.dataset)

        for batch_idx, (inputs,labels, y_) in enumerate(self.train_loader):
            inputs = inputs.to(ModelsConfig.device)
            labels = labels.to(ModelsConfig.device)

            self.optimizer.zero_grad()
            preds = self.model(inputs)
            
            loss = self.criterion(preds,labels)
            loss.backward()

            self.optimizer.step()
            train_loss.append(loss.item())
            # if batch_idx % verbose == 0 or batch_idx == len(self.train_loader)-1:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\telapsed: {:.2f} mins'.format(
            #         epoch, (batch_idx * len(inputs))+1, data_len, 100. * batch_idx / len(self.train_loader), loss.item(), (time()-start_time)/60))
                
        return loss
    def val_step(self, epoch):
        # print('\nevaluating…')
        self.model.eval()
        # test_loss = 0
        # mae = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels, y_) in enumerate(self.test_loader):
                if batch_idx==(len(self.test_loader)-1):
                    inputs = inputs.to(ModelsConfig.device)
                    labels = labels.to(ModelsConfig.device)

                    self.optimizer.zero_grad()

                    preds = self.model(inputs)

                    loss = self.criterion(preds,labels)

                    test_loss = loss.item()# / len(self.test_loader)
                    mae = self.eval_step(y_, preds)# / len(self.test_loader)

        # print('Test set: Average loss: {:.4f} | MAE: {:.4f}\n'.format(test_loss, mae))

        return test_loss, self.model.state_dict(), mae

    def eval_step(self, labels, preds):
        preds = preds.cpu().detach().numpy().reshape(-1, ModelsConfig.N_STEPS)
        # print(labels)
        y_true = self.target_scaler.inverse_transform(labels[0]['target'].values.reshape(-1, 1))
        y_pred = self.target_scaler.inverse_transform(preds[labels[0].sku_coded.values.astype(int), labels[0].time_step.values].reshape(-1, 1))

        mae = np.absolute(np.subtract(y_true, y_pred)).mean()/len(labels)

        return mae
    
    # def eval_step(self, labels: list, preds):
    #     mae = 0
    #     preds = preds.cpu().detach().numpy().reshape(len(labels), -1)
        
    #     # pandarallel.initialize(use_memory_fs=False)

    #     for i,label in enumerate(labels):
    #         pred_arr = preds[i].reshape(-1, N_STEPS, 6)
    #         label['y_true'] = label['sellin'].copy()
    #         label['y_pred'] = pred_arr[label.sku_coded.values, label.idx.values, 0]
            
    #         y_true = label['y_true'].values.reshape(1, -1)
    #         y_pred = self.target_scaler.inverse_transform(label['y_pred'].values.reshape(1, -1))
    #         label['y_pred'] = y_pred.reshape(-1,)
    #         mae += np.absolute(np.subtract(y_true, y_pred)).mean()/len(labels)

    #     return mae
