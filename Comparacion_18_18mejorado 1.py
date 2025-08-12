import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Ruta base para dataset
dataset_path = r"C:\UNIVERSIDAD\MAESTRIA EN ELECTRONICA DIGITAL\TOPICOS ESPECIALES II\proyecto machine learning\classification_dataset"

# Transformaciones idénticas para test
transform = transforms.Compose([
    transforms.Pad(padding=10, fill=0),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Carga dataset test
test_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# Función para evaluar modelo y obtener etiquetas + probabilidades para AUROC
def eval_model(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())  # Clase positiva probabilidad
    return np.array(all_labels), np.array(all_probs)

# Cargar modelos
def load_model(path_checkpoint, num_classes=2):
    model = models.resnet18(weights=None)  # Sin pesos pretrained para cargar custom
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(path_checkpoint, map_location=device))
    model.to(device)
    return model

# Paths de checkpoints
path_original = r"C:\UNIVERSIDAD\MAESTRIA EN ELECTRONICA DIGITAL\TOPICOS ESPECIALES II\proyecto machine learning\resnet18_original.pth"
path_mejorado = r"C:\UNIVERSIDAD\MAESTRIA EN ELECTRONICA DIGITAL\TOPICOS ESPECIALES II\proyecto machine learning\best_model.pth"

# Carga modelos
model_original = load_model(path_original)
model_mejorado = load_model(path_mejorado)

# Evalúa modelos
labels_orig, probs_orig = eval_model(model_original, test_loader, device)
labels_mej, probs_mej = eval_model(model_mejorado, test_loader, device)

# Calcular AUROC
auc_orig = roc_auc_score(labels_orig, probs_orig)
auc_mej = roc_auc_score(labels_mej, probs_mej)

print(f"AUROC ResNet18 original: {auc_orig:.4f}")
print(f"AUROC ResNet18 mejorado: {auc_mej:.4f}")

# Graficar
plt.figure(figsize=(8,5))
plt.bar(['ResNet18 Original', 'ResNet18 Mejorado'], [auc_orig, auc_mej], color=['orange', 'green'])
plt.ylim(0.5,1)
plt.ylabel('AUROC')
plt.title('Comparación AUROC entre modelos')
plt.show()
