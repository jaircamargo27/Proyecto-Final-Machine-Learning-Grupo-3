# --- 1. Importaciones ---
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# --- 2. Configuración del dispositivo ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# --- 3. Transformaciones para las imágenes ---
transform = transforms.Compose([
    transforms.Pad(padding=10, fill=0),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# --- 4. Carga de datasets ---
train_dataset = datasets.ImageFolder(root="classification_dataset/train", transform=transform)
val_dataset = datasets.ImageFolder(root="classification_dataset/val", transform=transform)
test_dataset = datasets.ImageFolder(root="classification_dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Clases:", train_dataset.classes)  # Muestra las clases, ej: ['background', 'defects']

# --- 5. Definición del modelo ---
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # Cambia 2 si tienes más clases
model = model.to(device)

# --- 6. Definir función de pérdida y optimizador ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 7. Función para entrenar por una época ---
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

# --- 8. Función para evaluar sin entrenar ---
def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_preds)

# --- 9. Ciclo de entrenamiento principal ---
num_epochs = 10  # Cambia la cantidad de épocas si quieres
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_labels, val_preds = eval_model(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")

# --- 10. Evaluación final en test y reporte ---
test_loss, test_labels, test_preds = eval_model(model, test_loader, criterion, device)
print("\nReporte clasificación en test:")
print(classification_report(test_labels, test_preds, target_names=train_dataset.classes))
print("Matriz de confusión:")
print(confusion_matrix(test_labels, test_preds))
