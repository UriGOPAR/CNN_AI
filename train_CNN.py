import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataloader import load_data
from model import MushroomCNN, ImprovedMushroomCNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando GPU" if torch.cuda.is_available() else "Usando CPU")

# Configura el directorio de datos
data_dir = "Mushrooms"  # Cambia a la ruta correcta de tus datos

# Cargar datos y clases
train_loader, val_loader, test_loader, classes = load_data(data_dir, batch_size=32)

# Selecciona el modelo a utilizar
model_version = "improved"  # Cambia a "original" o "improved" para seleccionar el modelo
if model_version == "original":
    model = MushroomCNN(num_classes=len(classes)).to(device)
elif model_version == "improved":
    model = ImprovedMushroomCNN(num_classes=len(classes)).to(device)

# Inicializar la función de pérdida
criterion = nn.CrossEntropyLoss()

# Inicializar el optimizador AdamW con un mayor weight decay para regularización L2
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)

# Configurar un programador de tasa de aprendizaje para reducir el learning rate cuando no mejora la pérdida de validación
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Almacenar los valores de pérdida y precisión por epoch
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Early stopping parameters
best_val_loss = float('inf')
patience = 10
counter = 0

# Training loop con early stopping
epochs = 200
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    train_loader_tqdm = tqdm(train_loader, desc=f"Entrenamiento Epoch {epoch+1}/{epochs}", unit="batch")

    for images, labels in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

        train_loader_tqdm.set_postfix(loss=loss.item())

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct_train / total_train)

    # Validation
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

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(100 * correct_val / total_val)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, Val Acc: {val_accuracies[-1]:.2f}%")

    # Actualizar el scheduler con la pérdida de validación
    scheduler.step(val_loss)

    # Implementación de Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping activado")
            break

print("Training complete.")

# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrix for Test Set
all_preds = []
all_labels = []

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

test_accuracy = 100 * correct_test / total_test
print(f"Test Accuracy: {test_accuracy:.2f}%")

conf_matrix = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=classes)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix')
plt.show()
