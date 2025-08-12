import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import time

def main():
    # --- 1. Configuración del dispositivo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

    # --- 2. Transformaciones para las imágenes ---
    transform = transforms.Compose([
        transforms.Pad(padding=10, fill=0),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # --- 3. Carga de datasets ---
    train_dataset = datasets.ImageFolder(root="classification_dataset/train", transform=transform)
    val_dataset = datasets.ImageFolder(root="classification_dataset/val", transform=transform)
    test_dataset = datasets.ImageFolder(root="classification_dataset/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Clases:", train_dataset.classes)

    # --- 4. Definición del modelo ResNet101 ---
    model = models.resnet101(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(train_dataset.classes))  # Ajustar para el número de clases detectadas
    model = model.to(device)

    # --- 5. Función de pérdida y optimizador ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 6. Funciones para entrenamiento y evaluación ---
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
        all_preds = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        return running_loss / total, correct / total, np.array(all_labels), np.array(all_preds)

    # --- 7. Entrenamiento principal ---
    num_epochs = 20
    train_losses, val_losses = [], []
    train_accs = []
    val_accs = []
    times = []

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = eval_model(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_time = end_time - start_time
        times.append(epoch_time)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f} - "
              f"Tiempo: {epoch_time:.2f} seg")

    # --- 8. Evaluación final en test ---
    test_loss, test_acc, test_labels, test_preds = eval_model(model, test_loader, criterion, device)

    print("\nReporte clasificación en test:")
    print(classification_report(test_labels, test_preds, target_names=train_dataset.classes))

    print("Matriz de confusión:")
    print(confusion_matrix(test_labels, test_preds))

    print(f"Test accuracy: {test_acc:.4f}")

    # --- 9. Gráficas ---
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Pérdida de entrenamiento y validación')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.title('Precisión de entrenamiento y validación')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, times, label='Tiempo por época (seg)')
    plt.xlabel('Épocas')
    plt.ylabel('Segundos')
    plt.title('Tiempo por época')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
