# dataloader.py

import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Definir la transformación en una función separada
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

class MyImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super(MyImageFolder, self).__getitem__(index)
        except OSError as e:
            print(f"Imagen dañada en el índice {index}, saltando.")
            return None

def load_data(data_dir, batch_size=16):
    transform = get_transform() 
    
    dataset = MyImageFolder(root=data_dir, transform=transform)

    train_val_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_val_size
    train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])

    train_size = int(0.75 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset.classes
