import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from DataClass import DataClass
from rnnClass import RNN
from sklearn.model_selection import train_test_split
import datetime
import os
from mpl_toolkits.mplot3d import Axes3D

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'models/model.pth')
file_path = os.path.join(script_dir, 'data_array_smaller.csv')

# Load and setup model
model = torch.load('models/model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
# Load data
data = pd.read_csv(file_path, nrows=100000)
print(data.head())
Data = DataClass(data)

# Run inference
points = []
done = False
while not done:
    run, done = Data.next()
    for point_num in range(1):  # Currently only processing first point
        single_point = run[point_num].unsqueeze(0)  # Unsqueeze to add batch dimension
        print(single_point)
        single_point = single_point.to(device)
        output = model(single_point)
        points.append(output)
    done = True

# Extract coordinates
x_coords = []
y_coords = []
z_coords = []
for output in points:
    x_coords.append(output[0][0].item())  # X coordinate
    y_coords.append(output[0][1].item())  # Y coordinate
    z_coords.append(output[0][2].item())  # Z coordinate

# Convert to numpy arrays
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)
z_coords = np.array(z_coords)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')

# Set labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Scatter Plot of X, Y, Z Points')

plt.show()
