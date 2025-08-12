import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

# --- Configuración dispositivo ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# --- Transformaciones ---
transform = transforms.Compose([
    transforms.Pad(padding=10, fill=0),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Carga datasets ---
dataset_path = r"C:\UNIVERSIDAD\MAESTRIA EN ELECTRONICA DIGITAL\TOPICOS ESPECIALES II\proyecto machine learning\classification_dataset"
train_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, "train"), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, "val"), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Clases:", train_dataset.classes)

# --- Balanceo de clases para CrossEntropyLoss ---
class_counts = np.bincount([label for _, label in train_dataset.imgs])
class_weights = 1. / (class_counts + 1e-6)  # evitar división por 0
class_weights = class_weights / class_weights.sum()
weights_tensor = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# --- Modelo ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# --- Funciones para entrenamiento y evaluación ---
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)  # probabilidades
            preds = probs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())  # prob clase positiva

    avg_loss = running_loss / total
    accuracy = correct / total
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = None  # no se pudo calcular AUC (e.g. solo una clase en labels)

    return avg_loss, accuracy, auc, all_labels, all_probs

# --- Entrenamiento principal ---
num_epochs = 20
best_val_auc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []
val_aucs = []

for epoch in range(num_epochs):
    start_time = time.time()
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_auc, _, _ = eval_model(model, val_loader, criterion, device)
    scheduler.step()
    end_time = time.time()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    val_aucs.append(val_auc if val_auc is not None else 0)

    if val_auc is not None and val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - "
          f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, Val AUC: {val_auc:.4f} - "
          f"Tiempo: {(end_time - start_time):.2f} seg")

# --- Evaluación test ---
model.load_state_dict(torch.load("best_model.pth"))
test_loss, test_acc, test_auc, test_labels, test_probs = eval_model(model, test_loader, criterion, device)

test_preds = (test_probs >= 0.5).astype(int)  # umbral 0.5 para clases

print("\nReporte clasificación en test:")
print(classification_report(test_labels, test_preds, target_names=train_dataset.classes))
print("Matriz de confusión:")
print(confusion_matrix(test_labels, test_preds))
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# --- Gráficas ---
import matplotlib.pyplot as plt
epochs = range(1, num_epochs+1)
plt.figure(figsize=(14,5))

plt.subplot(1,3,1)
plt.plot(epochs, train_losses, label='Pérdida entrenamiento')
plt.plot(epochs, val_losses, label='Pérdida validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida')

plt.subplot(1,3,2)
plt.plot(epochs, train_accs, label='Precisión entrenamiento')
plt.plot(epochs, val_accs, label='Precisión validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.title('Precisión')

plt.subplot(1,3,3)
plt.plot(epochs, val_aucs, label='AUC validación')
plt.xlabel('Épocas')
plt.ylabel('AUC')
plt.legend()
plt.title('AUROC Validación')

plt.tight_layout()
plt.show()
