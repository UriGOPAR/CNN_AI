# predict.py

import torch
from model import ImprovedMushroomCNN
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from dataloader import get_transform  # Importar la función de transformación

# Cargar las clases desde el archivo
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f]

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Función para cargar y preprocesar la imagen usando la transformación importada
def preprocess_image(image):
    transform = get_transform()  # Utiliza la misma transformación que en el DataLoader
    return transform(image)

# Cargar el modelo
model = ImprovedMushroomCNN(num_classes=len(classes))
model.load_state_dict(torch.load("mushroom_model.pth"))
model.to(device)
model.eval()

# Función para hacer predicciones en múltiples imágenes y mostrarlas
def predict_and_display_images(image_paths):
    for image_path in image_paths:
        # Cargar y preprocesar la imagen
        image = Image.open(image_path)
        preprocessed_image = preprocess_image(image)
        preprocessed_image = preprocessed_image.unsqueeze(0)  # Añadir una dimensión para el batch

        # Enviar la imagen al dispositivo y hacer predicción
        preprocessed_image = preprocessed_image.to(device)
        with torch.no_grad():
            outputs = model(preprocessed_image)
            _, predicted = torch.max(outputs, 1)
            prediction = classes[predicted.item()]

        # Mostrar la imagen y la predicción
        plt.imshow(image)
        plt.title(f"Predicción: {prediction}")
        plt.axis('off')  
        plt.show()

# Ejemplo de uso
image_paths = ["predecir/cha.jpg"]
predict_and_display_images(image_paths)
