import os, sys
import torch
import torch as tc
from torch.distributions import Normal
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import util

import pandas as pd
from model.pred_set import PredSet, PredSetReg
from learning import PredSetConstructor
from main_cls import parse_args
from model.util import neg_log_prob

class BoundingBoxttDataset(Dataset):
    def __init__(self, data_dir, data_names, transform=None):
        data = []
        for d in data_names:
            d_dir = os.path.join(data_dir, d)
            data.append(torch.tensor(pd.read_csv(d_dir, header=None).values)[:, 1:13])
        self.data = torch.vstack(data)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        preds= self.data[idx, :8]
        gts = self.data[idx, 8:]
        if self.transform:
            preds = self.transform(preds)
        return preds, gts

class logprob_mdl:
    def __init__(self):
        print("Instance Created")
      
    # Defining __call__ method
    def __call__(self, x, y):
        mu = x[:, :4]
        logvar = x[:, 4:]
        return {'logph': -neg_log_prob(yhs=mu, yhs_logvar=logvar, ys=y)}
    
    def forward(self, x):
        return {'mu': x[:, :4], 'logvar': x[:, 4:]} 

args = parse_args()
  
# Instance created
mdl = logprob_mdl()

training_data = BoundingBoxttDataset(data_dir='~/tracking/tracking_wo_bnw/test_results_wo_KF', data_names=["MOT17-02-FRCNN_data_wo_KF.csv", "MOT17-04-FRCNN_data_wo_KF.csv"] )
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
mdl_predset = PredSetReg(mdl, eps=1e-1, delta=1e-5, n=len(training_data))
l = PredSetConstructor(mdl_predset, params=args.train_predset)
testing_data = BoundingBoxttDataset(data_dir='~/tracking/tracking_wo_bnw/test_results_wo_KF', data_names=["MOT17-05-FRCNN_data_wo_KF.csv", "MOT17-09-FRCNN_data_wo_KF.csv"] )
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)
l.train(train_dataloader)
l.test(test_dataloader, ld_name="without KF", verbose=True)