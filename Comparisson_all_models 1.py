import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuración dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformaciones igual que en entrenamiento
transform = transforms.Compose([
    transforms.Pad(padding=10, fill=0),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Carga dataset test (ruta igual que antes)
dataset_path = r"C:\UNIVERSIDAD\MAESTRIA EN ELECTRONICA DIGITAL\TOPICOS ESPECIALES II\proyecto machine learning\classification_dataset"
test_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Función para crear modelo y cargar pesos
def load_model(pth_path):
    model = models.resnet18(weights=None)  # Sin pesos por defecto para evitar warning
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(test_dataset.classes))
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Función para obtener probabilidades y etiquetas reales
def get_probs_labels(model, loader):
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probabilidad clase positiva
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_probs)

# Cargar modelos
model_paths = {
    "model": "model.pth",
    "best_model": "best_model.pth",
    "best_model_improved": "best_model_improved.pth"
}

results = {}

for name, path in model_paths.items():
    if os.path.exists(path):
        model = load_model(path)
        labels, probs = get_probs_labels(model, test_loader)
        auc = roc_auc_score(labels, probs)
        fpr, tpr, _ = roc_curve(labels, probs)
        results[name] = (auc, fpr, tpr)
        print(f"{name} - AUROC: {auc:.4f}")
    else:
        print(f"Archivo {path} no encontrado.")

# Graficar curvas ROC
plt.figure(figsize=(8,6))
for name, (auc, fpr, tpr) in results.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

plt.plot([0,1], [0,1], 'k--', label="Aleatorio")
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Comparación Curvas ROC Modelos")
plt.legend()
plt.grid(True)
plt.show()
