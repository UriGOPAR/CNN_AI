import torch
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt
from PIL import ImageFile

# Forzar a PIL a cargar imágenes truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set the device (GPU with CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset from the folder structure
data_dir = "Mushrooms"  # Replace with the actual path
# Clase personalizada para manejar imágenes dañadas
class MyImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super(MyImageFolder, self).__getitem__(index)
        except OSError as e:
            print(f"Imagen dañada en el índice {index}, saltando.")
            return None

# Define transforms for preprocessing (resize, normalization, etc.)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a fixed size
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
])

dataset = ImageFolder(root=data_dir, transform=transform)

# Shuffle and split the dataset (60% train, 20% val, 20% test)
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoader for each split
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define a simple CNN architecture
class MushroomCNN(torch.nn.Module):
    def __init__(self):
        super(MushroomCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(64 * 32 * 32, 512)
        self.fc2 = torch.nn.Linear(512, len(dataset.classes))  # Output layer, number of mushroom types
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)  # Flatten the feature map
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the CNN model, loss function, and optimizer
model = MushroomCNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Para almacenar las métricas de cada época
train_losses = []
val_losses = []
test_losses = []
train_accuracies = []
val_accuracies = []
test_accuracies = []

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Entrenamiento
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Clear the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct_train / total_train)
    
    # Validación
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(100 * correct_val / total_val)

    # Evaluación en el conjunto de test durante cada época
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_losses.append(test_loss / len(test_loader))
    test_accuracies.append(100 * correct_test / total_test)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, Val Acc: {val_accuracies[-1]:.2f}%, Test Acc: {test_accuracies[-1]:.2f}%")

print("Training complete.")

# Graficar el rendimiento
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Val Loss')
plt.plot(epochs_range, test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Loss vs Epochs')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
plt.plot(epochs_range, val_accuracies, label='Val Accuracy')
plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy vs Epochs')

plt.tight_layout()
plt.show()

# Evaluar el modelo en el conjunto de prueba y mostrar las imágenes
model.eval()
correct = 0
total = 0
images_shown = 0
fig = plt.figure(figsize=(15, 15))

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Mostrar las primeras 30 imágenes con sus predicciones
        for i in range(min(30 - images_shown, images.size(0))):
            img = images[i].cpu().permute(1, 2, 0)  # Cambiar el formato a (H, W, C)
            img = img * 0.5 + 0.5  # Desnormalizar la imagen
            
            ax = fig.add_subplot(6, 5, images_shown + 1, xticks=[], yticks=[])
            ax.imshow(img)
            ax.set_title(f"Pred: {dataset.classes[predicted[i]]}\nTrue: {dataset.classes[labels[i]]}")
            images_shown += 1

        if images_shown >= 30:
            break

# Mostrar las imágenes
plt.tight_layout()
plt.show()

print(f"Test Accuracy: {100 * correct / total}%")
