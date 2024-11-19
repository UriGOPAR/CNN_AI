import os
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

data_dir = "Mushrooms"
train_loader, val_loader, test_loader, classes = load_data(data_dir, batch_size=32)

model_version = "improved"
if model_version == "original":
    model = MushroomCNN(num_classes=len(classes)).to(device)
elif model_version == "improved":
    model = ImprovedMushroomCNN(num_classes=len(classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
 
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop sin early stopping
epochs = 40
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

    scheduler.step(val_loss)

print("Training complete.")

# Graficar pérdidas
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Graficar precisión
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Calcular matriz de confusión
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

# Guardar el modelo entrenado (con reemplazo)
model_path = "mushroom_model.pth"
if os.path.exists(model_path):
    os.remove(model_path)
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en {model_path}")

# Guardar clases en "classes.txt" (con reemplazo)
class_file = "classes.txt"
if os.path.exists(class_file):
    os.remove(class_file)
with open(class_file, "w") as f:
    for class_name in classes:
        f.write(f"{class_name}\n")
print(f"Clases guardadas en {class_file}")
