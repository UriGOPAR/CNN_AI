import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt
from PIL import ImageFile
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from tqdm import tqdm 

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set the device (GPU with CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Verificar si se está utilizando la GPU
if torch.cuda.is_available():
    print("Usando GPU:", torch.cuda.get_device_name(0))
else:
    print("Usando CPU")

# Load the dataset from the folder structure
data_dir = "Mushrooms"  # Replace with the actual path

# Clase personalizada para manejar imágenes dañadas
class MyImageFolder(ImageFolder):
    def __getitem__(self, index):
        # Intentar cargar la imagen y manejar la excepción en caso de error
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

# Almacenar los valores de pérdida y precisión por epoch
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Barra de progreso para el entrenamiento
    train_loader_tqdm = tqdm(train_loader, desc=f"Entrenamiento Epoch {epoch+1}/{epochs}", unit="batch")

    for images, labels in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights

        running_loss += loss.item()

        # Calcular precisión en el entrenamiento
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

        # Actualizar el texto de la barra de progreso
        train_loader_tqdm.set_postfix(loss=loss.item())

    # Guardar pérdida y precisión de entrenamiento
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct_train / total_train)

    # Evaluación en el conjunto de validación con barra de progreso
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

            _, predicted_val = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted_val == labels).sum().item()

    # Guardar pérdida y precisión de validación
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(100 * correct_val / total_val)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, Val Acc: {val_accuracies[-1]:.2f}%")

print("Training complete.")

# Gráfica de pérdida de entrenamiento y validación por época
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)  # Agregar cuadrícula para mejorar la visibilidad
plt.show()

# Gráfica de precisión para validación y entrenamiento por época
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy per Epoch')
plt.legend()
plt.show()

# Matriz de confusión en el conjunto de prueba
all_preds = []
all_labels = []

# Obtener predicciones y etiquetas verdaderas en el conjunto de prueba
model.eval()
with torch.no_grad():
    correct_test = 0
    total_test = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

# Calcular y mostrar la precisión en el conjunto de prueba
test_accuracy = 100 * correct_test / total_test
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Crear y mostrar la matriz de confusión
conf_matrix = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=dataset.classes)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix')
plt.show()

# Mostrar 40 imágenes con predicciones y etiquetas reales
fig = plt.figure(figsize=(20, 20))
images_shown = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Mostrar las primeras 40 imágenes
        for i in range(min(40 - images_shown, images.size(0))):
            img = images[i].cpu().permute(1, 2, 0)  # Cambiar el formato a (H, W, C)
            img = img * 0.5 + 0.5  # Desnormalizar la imagen
            
            ax = fig.add_subplot(8, 5, images_shown + 1, xticks=[], yticks=[])
            ax.imshow(img)
            ax.set_title(f"Pred: {dataset.classes[predicted[i]]}\nTrue: {dataset.classes[labels[i]]}")
            images_shown += 1

        if images_shown >= 40:
            break

plt.tight_layout()
plt.show()
