import torch
#from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import random
from PIL import Image
from matplotlib import pyplot as plt
from Utilities import *
script_dir = os.path.dirname(os.path.abspath(__file__))
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit


# Load the entire model (architecture + weights)
model_path = os.path.join(script_dir, 'models/not_a_good_run.pth')
model = torch.load(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

##Load in the validation starting data
file_path = os.path.join(script_dir, 'data_array_smaller.csv')
data = pd.read_csv(file_path, nrows=100000)
print(data.head())
Data = DataClass(data)


##Run inference model on these

##run the model for one segment
i = 0
points = []
done = False

while not done:
    run, done = Data.next()   
    #for point_num in range(len(run)):
    for point_num in range(1):
        single_point = run[point_num].unsqueeze(0)  # Unsqueeze to add batch dimension
        print(single_point)
        single_point = single_point.to(device)
        output = model(single_point)  
        points.append(output)
    done = True
    #print(points)
x_coords = []
y_coords = []
z_coords = []

# Extract the coordinates from the model outputs
for output in points:
    # Assuming output is of shape [1, 3] where [X, Y, Z] are in the first three columns
    # If your output shape is different, adjust the indices accordingly
    x_coords.append(output[0][0].item())  # X coordinate
    y_coords.append(output[0][1].item())  # Y coordinate
    z_coords.append(output[0][2].item())  # Z coordinate

# Convert lists to numpy arrays (optional, but helpful for plotting)
import numpy as np
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)
z_coords = np.array(z_coords)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')  # You can change color and marker

# Set labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Scatter Plot of X, Y, Z Points')

# Show the plot
plt.show()