# Proyecto-Final-Machine-Learning-Grupo-3
Clasificación de Defectos en Infraestructuras con Modelos Preentrenados utilizando el dataset CODEBRIM (Concrete Defect BRidge IMage Dataset), diseñado para tareas de clasificación de imágenes con aplicación en infraestructura civil.
Este proyecto desarrollado por Jair Camargo y Jorge Espinoza implementa y compara tres modelos de redes neuronales convolucionales preentrenadas (ResNet18, ResNet50 y ResNet101) para la detección de defectos en imágenes del dataset CODEBRIM. El objetivo es clasificar imágenes en dos clases: `defects` y `background`.

## 📁 Estructura del Dataset

- **Entrenamiento**: 4,297 imágenes con defectos, 2,186 sin defectos
- **Validación**: 467 con defectos, 150 sin defectos
- **Prueba**: 483 con defectos, 150 sin defectos

Las imágenes se preprocesan con padding y redimensionamiento a 224x224 píxeles.

## 🧠 Modelos Utilizados

- `ResNet18`: Modelo base para validar el pipeline
- `ResNet50`: Modelo intermedio con mayor capacidad
- `ResNet101`: Modelo profundo para extracción avanzada de características

Todos los modelos se cargan con pesos preentrenados en ImageNet y se ajusta la capa final para clasificación binaria.

## 📊 Métricas Evaluadas

- **Accuracy**: Precisión global
- **Precision**: Fiabilidad al detectar defectos
- **Recall**: Capacidad de detectar todos los defectos reales
- **F1-score**: Equilibrio entre precision y recall
- **AUC ROC**: Área bajo la curva ROC para evaluar discriminación entre clases

## 🧪 Resultados Obtenidos

Los modelos alcanzaron altos niveles de precisión, con ResNet18 mejorado logrando:
- Accuracy: 96.5%
- AUC: 0.9858
- F1-score: 0.97

## ⚙️ Instrucciones de Uso

1. Clonar el repositorio y colocar el dataset en la carpeta `classification_dataset`.
2. Ejecutar el script correspondiente al modelo deseado (`ResNet18.py`, `ResNet50.py`, `ResNet101.py`).
3. Los resultados se mostrarán en consola e incluirán métricas y matriz de confusión.

## 📦 Dependencias

- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- matplotlib

## 📌 Notas Adicionales

El proyecto incluye técnicas de mejora como balanceo de clases, uso de AUC como métrica principal, y visualización de curvas ROC para comparar modelos.

---
