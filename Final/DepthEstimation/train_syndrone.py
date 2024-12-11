from syndrone_utilities import *

import os
import glob
import torch
import cv2
import argparse
from matplotlib import pyplot as plt
from datetime import datetime

import DPT.util.io as io

from torchvision.transforms import Compose

from DPT.dpt.models import DPTDepthModel
from DPT.dpt.midas_net import MidasNet_large
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Starting run: {run_time}")
os.makedirs(f"DepthEstimation/models/{run_time}", exist_ok=True)

writer = SummaryWriter(f'runs/{run_time}')

# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: %s" % device)

hparams = {
    'learning_rate':1e-4,
    'batch_size': 1,
    'num_epochs': 30,
    'model_weights': "dpt_large",
    'lam': 0.5,
    'gamma': 0.94,
    'weight_decay': 0.005
}

writer.add_hparams(hparams, {})

# load model
model = load_model(weights=hparams['model_weights'],
                   device=device,
                   eval=False)

# Load dataloaders
dataloader_train = SyndroneDataloader(batch_size=1,shuffle=True, split='train')
dataloader_test = SyndroneDataloader(batch_size=1,shuffle=False, split='test')

# Optimizer and LR Schuduler
optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'], weight_decay=hparams['weight_decay'])
scheduler = ExponentialLR(optimizer, gamma=hparams['gamma'])

num_epochs = hparams['num_epochs']

debug = False
for epoch in range(num_epochs):
    epoch_start = datetime.now()
    print(f"Epoch {epoch}/{num_epochs}")

    # Training
    train_epoch_loss = 0
    model.train()
    for batch_idx, (inputs, truths) in enumerate(dataloader_train):

        optimizer.zero_grad()

        inputs, truths = inputs.to(device), truths.to(device)
        outputs = model(inputs)
        aligned_pred = align_pred(outputs.cpu())

        loss = eigen_loss(aligned_pred.cpu(), truths.cpu(), lam=hparams['lam'])
        train_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Truth = 1/distance (units)
        # Outputs = raw model output
        # Aligned Outputs = align_pred(output) = pred 1/distance
        # Pred distance = 1 / aligned_prediction

        writer.add_scalar('BatchLoss/train', loss.item(), epoch * len(dataloader_train) + batch_idx)

    train_epoch_loss = train_epoch_loss/len(dataloader_train)
    
    # Testing
    test_epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, truths) in enumerate(dataloader_test):

            inputs, truths = inputs.to(device), truths.to(device)
            outputs = model(inputs)
            aligned_pred = align_pred(outputs.cpu())

            loss = eigen_loss(aligned_pred.cpu(), truths.cpu(), lam=hparams['lam'])
            test_epoch_loss += loss.item()

            writer.add_scalar('BatchLoss/test', loss.item(), epoch * len(dataloader_test) + batch_idx)

    test_epoch_loss = test_epoch_loss / len(dataloader_test)

    epoch_runtime = datetime.now() - epoch_start
    print(f"\tEpoch Loss Train: {train_epoch_loss:.5f} Test: {test_epoch_loss:.5f} Epoch Time: {epoch_runtime.seconds // 60}min {epoch_runtime.seconds % 60}sec")

    print(scheduler.get_last_lr())
    writer.add_scalar('Loss/train', train_epoch_loss, epoch)
    writer.add_scalar('Loss/test', test_epoch_loss, epoch)
    writer.add_scalar('epoch_runtime', epoch_runtime.seconds, epoch)
    writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)

    torch.save(model.cpu().state_dict(), f"DepthEstimation/models/{run_time}/syndrone_weights_{epoch}.pt")
    model.to(device)
    scheduler.step()

torch.save(model.cpu().state_dict(), f"DepthEstimation/models/{run_time}/syndrone_weights.pt")
print("finished")
