import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

# Load resnet 50 best weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# # Freeze all but fc layer
# for name, param in model.named_parameters():
#     if "fc" not in name:
#         param.requires_grad = False

# change FC output to 10 classes
model.fc = nn.Linear(model.fc.in_features, 10)

# Hyper Parameters
hyperParams = {
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 0.001
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=hyperParams['batch_size'], shuffle=True)

test_dataset = datasets.ImageFolder(root='data/val', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=hyperParams['batch_size'], shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=hyperParams['learning_rate'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.20)

writer = SummaryWriter(log_dir='runs/testing')

try:
    for epoch in range(hyperParams['num_epochs']):
        model.train()
        train_avg_loss = 0.0
        total = 0
        correct = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            train_avg_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_avg_loss = train_avg_loss / len(train_loader)
        writer.add_scalar('Loss/train', train_avg_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        # print(f"Epoch [{epoch + 1}/{hyperParams['num_epochs']}], Loss: {running_loss / 10:.4f}")
        running_loss = 0.0

        model.eval()
        test_avg_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_avg_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_avg_loss = test_avg_loss / len(test_loader)
        writer.add_scalar('Loss/test', test_avg_loss, epoch)
        test_accuracy = 100 * correct / total
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        
        print(f'Test Loss: {test_avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
except KeyboardInterrupt:
    print('\nTraining Interrupted')

# Save Model
model_name = input('Enter name to save model, enter otherwise: ')
if model_name:
    torch.save(model, 'models/' + model_name + '.pth')
    print(f"Model saved in models/{model_name}.pth")