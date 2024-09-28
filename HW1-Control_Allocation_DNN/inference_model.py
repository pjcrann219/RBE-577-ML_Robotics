import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from models import *

def inference(model_name='test', data_name = 'data_6'):

    model = allocationNN(dropout=0.0)
    model.load_state_dict(torch.load('models/' + model_name + '.pt')['model_state_dict'])
    hyperParams = torch.load('models/' + model_name + '.pt')['hyperParams']
    model.eval()

    # Prepare input data for inference
    # Here you need to load or prepare the input data as a tensor
    input_data = np.load('data/' + data_name +'.npy')  # Replace with your input data path
    test_length = 1000

    input_u = torch.tensor(input_data[:5,:test_length], dtype=torch.float32).T
    input_tau = torch.tensor(input_data[5:,:test_length], dtype=torch.float32).T

    # input_tau = (input_tau - hyperParams['train_tau_mean']) / hyperParams['train_tau_std']


    with torch.no_grad():
        u, tau_rec = model(input_tau)


# Plotting
    fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
    column_labels = ['F1', 'F2', 'F3', 'Angle 2', 'Angle 3']

    for i in range(5):
        axes[i].plot(input_u[:, i], '-', label='data')
        axes[i].plot(u[:, i], '-', label='NN u')
        
        axes[i].legend()
        if i < 3:
            axes[i].set_ylabel(f'{column_labels[i]} (Force')
        else:
            axes[i].set_ylabel(f'{column_labels[i]} (Angle)')
        axes[i].set_title(f'Comparison of {column_labels[i]}')

    axes[-1].set_xlabel('Time')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    inference()