# Clasificacion-retina-CNN

## Descripción

Proyecto de minería de imágenes médicas enfocado en el análisis de imágenes OCT de retina. Se implementa extracción de características de textura mediante Gray Level Co-occurrence Matrix (GLCM), clasificación de patologías retinales utilizando Transfer Learning con MobileNetV2 y recuperación de imágenes similares mediante un sistema CBIR.

La aplicación cuenta con una interfaz gráfica desarrollada en Streamlit que permite al usuario cargar una imagen OCT, obtener un diagnóstico con probabilidades asociadas y visualizar las imágenes más similares recuperadas desde la base de datos.


## Características

- Clasificación automática de imágenes OCT.
- Implementación de Transfer Learning con MobileNetV2.
- Fine-Tuning de las últimas capas del modelo.
- Recuperación de imágenes similares mediante descriptores GLCM.
- Interfaz interactiva desarrollada con Streamlit.
- Visualización de probabilidades por clase.
- Recuperación configurable mediante diferentes métricas de distancia:
  - Euclidiana
  - Manhattan
  - Coseno


## Enfermedades consideradas

El sistema clasifica las siguientes categorías:

| Clase | Descripción |
|---------|------------|
| AMD | Age-related Macular Degeneration |
| CNV | Choroidal Neovascularization |
| CSR | Central Serous Retinopathy |
| DME | Diabetic Macular Edema |
| DR | Diabetic Retinopathy |
| DRUSEN | Drusen |
| MH | Macular Hole |
| NORMAL | Retina sin patología |


## Arquitectura del Sistema

### Módulo de Clasificación

- Modelo base: MobileNetV2
- Pesos iniciales: ImageNet
- Entrada: 224 × 224 píxeles
- Función de activación final: Softmax
- Número de clases: 8

### Módulo CBIR

Para cada imagen se calcula un descriptor GLCM compuesto por:

- Contraste
- Homogeneidad
- Energía
- Correlación
- Angular Second Moment (ASM)

Estos descriptores son almacenados y utilizados para recuperar las imágenes más similares a la imagen de consulta.
