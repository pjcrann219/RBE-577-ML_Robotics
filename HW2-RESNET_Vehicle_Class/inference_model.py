import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import random
from PIL import Image
from matplotlib import pyplot as plt

# Load the entire model (architecture + weights)
model = torch.load('models/full_retrain_scheduler.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Get list of class names (folder names)
class_names = os.listdir('data/val')
class_names.sort()
class_index_to_name = {index: name for index, name in enumerate(class_names)}

labels = [f for f in os.listdir('data/val')]

input_tensors = []

for label in labels:
    image_name = random.choice(os.listdir('data/val/' + label))
    image_path = 'data/val/' + label + '/' + image_name
    image = Image.open(image_path)
    image_tensor = transform(image)
    input_tensors.append(image_tensor.unsqueeze(0))

input_tensor_batch = torch.cat(input_tensors, dim=0)

# Run inference on these inputs
outputs = model(input_tensor_batch.to(device))
_, predicted_classes = torch.max(outputs, 1)
predicted_labels = [class_index_to_name[int(k)] for k in predicted_classes.cpu()]

# Display truth vs pred
print(labels)
print(predicted_labels)

# Plot images with labels as example
fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # Create a 2x5 grid of subplots
axes = axes.flatten()  # Flatten the axes array for easy iteration

for ax, img, true_label, pred_label in zip(axes, input_tensor_batch, labels, predicted_labels):
    img = img.permute(1, 2, 0)  # Rearrange the tensor to (H, W, C) for plotting
    img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # Denormalize
    img = img.clamp(0, 1)  # Ensure pixel values are in [0, 1]
    
    ax.imshow(img.numpy())
    ax.axis('off')
    ax.set_title(f'True: {true_label}\nPred: {pred_label}')

plt.tight_layout()
plt.show()