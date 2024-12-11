import torch
from matplotlib import pyplot as plt
from sklearn.linear_model import HuberRegressor
import numpy as np

from DPT.dpt.models import DPTDepthModel
from syndrone_utilities import *

def get_scale_shift(pred, truth):
    # Given pred and truth depth maps, calculate 
    pred_flat = pred.flatten()
    truth_flat = truth.flatten()
    valid = ~torch.isnan(truth_flat)
    
    # Get valid pixels only
    pred_valid = pred_flat[valid]
    truth_valid = truth_flat[valid]
    
    # Fit robust regression
    huber = HuberRegressor()
    huber.fit(pred_valid.reshape(-1, 1), truth_valid)
    
    return huber.coef_[0], huber.intercept_

optimize = False

# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("device: %s" % device)

# load network
model_type = "dpt_large" # DPT-Large
model_path = "DepthEstimation/DPT/weights/dpt_large-midas-2f21e586.pt"

model = DPTDepthModel(
    path=model_path,
    backbone="vitl16_384",
    non_negative=True,
    enable_attention_hooks=False,
)

if optimize == True and device == torch.device("cuda"):
    model = model.to(memory_format=torch.channels_last)
    model = model.half()

model.to(device)
model.eval()

print(f"{model_type} loaded from {model_path}")

dataloader = SyndroneDataloader()

ms = []
bs = []

for batch_idx, (input, truth) in enumerate(dataloader):

    with torch.no_grad():
        input, truth = input.to(device), truth.to(device)
        output = model(input)
        
        m, b = get_scale_shift(output.cpu(), truth.cpu())
        ms.append(m)
        bs.append(b)
        print(f"batch: {batch_idx}/{len(dataloader)} m = {m:.9f} b = {b:.9f}")

avg_m, avg_b = sum(ms) / len(ms), sum(bs) / len(bs)
print(f"avg_m = {avg_m:.9f}, avg_b = {avg_b:.9f}")