# Reporte Primera Entrega: Sistema de Anotación de Video con IA

**Collin Gonzalez - A00382429**

**Juan Felipe Ramirez -**


## 1. Preguntas de Interés

### Pregunta Principal
¿Cómo podemos desarrollar un sistema preciso y en tiempo real que detecte y clasifique automáticamente diferentes actividades físicas básicas de una persona a partir de video?

### Preguntas Secundarias
- ¿Qué características de los movimientos son más relevantes para distinguir entre diferentes actividades?
- ¿Cómo podemos normalizar los datos de pose para que sean independientes de la altura de la persona y la distancia a la cámara?
- ¿Qué tan robusto es el sistema ante diferentes ángulos de cámara y condiciones de iluminación?

## 2. Tipo de Problema

Este es un problema de clasificación multiclase en tiempo real, específicamente:
- **Entrada**: Secuencia de frames de video
- **Salida**: Clasificación entre 7 clases de actividades:
  - Caminando de frente
  - Caminando de espaldas
  - Caminando de lado
  - Giro
  - Parado
  - Sentado
  - Quieto

## 3. Metodología

Seguimos la metodología CRISP-DM adaptada a nuestro contexto:

1. **Comprensión del Negocio**
   - Identificación de actividades clave a detectar
   - Definición de requisitos de tiempo real
   - Establecimiento de métricas de éxito

2. **Comprensión de los Datos**
   - Recopilación de videos de diferentes personas
   - Extracción de landmarks usando MediaPipe
   - Análisis de la calidad de detección de poses

3. **Preparación de los Datos**
   - Extracción de landmarks de pose (33 puntos)
   - Normalización de coordenadas
   - Generación de dataset estructurado

4. **Modelado** (próximas entregas)
   - Selección de algoritmos
   - Entrenamiento y validación
   - Optimización de hiperparámetros

5. **Evaluación** (próximas entregas)
   - Validación con métricas definidas
   - Pruebas en tiempo real
   - Análisis de errores

6. **Implementación** (próximas entregas)
   - Desarrollo de interfaz
   - Optimización de rendimiento
   - Documentación

## 4. Métricas de Evaluación

### Métricas de Clasificación
- Accuracy global
- Precision y Recall por clase
- F1-Score
- Matriz de confusión

### Métricas de Rendimiento

- Frames por segundo (FPS)
- Latencia de detección
- Uso de recursos (CPU/RAM)

## 5. Datos Recolectados

### Dataset Actual

- **Número de videos**: [X] videos
- **Duración total**: [Y] minutos
- **Número de personas**: [Z] personas diferentes
- **Distribución de clases**:
  - Caminando de frente: [%]
  - Caminando de espaldas: [%]
  - [etc...]

### Proceso de Recolección

1. Grabación de videos con diferentes personas
2. Procesamiento con MediaPipe para extracción de poses
3. Generación de CSV con landmarks normalizados
4. Combinación en dataset unificado

## 6. Análisis Exploratorio de Datos

### Características Extraídas

- 33 puntos de referencia (landmarks)
- 3 coordenadas por punto (x, y, z)
- Total de 99 características por frame

## 7. Estrategias para Aumentar el Dataset

1. **Data Augmentation**
   - Rotación de poses
   - Escalado de coordenadas
   - Variación de velocidades

2. **Colaboración Externa**
   - Colaboración con otros grupos
   - Grabación de más voluntarios
   - Intercambio de datos anonimizados

## 8. Aspectos Éticos

### Privacidad y Consentimiento
- Obtención de consentimiento informado
- Anonimización de datos
- Almacenamiento seguro de información

### Transparencia
- Documentación clara del proceso
- Explicabilidad de las decisiones del modelo
- Limitaciones conocidas

## 9. Próximos Pasos

1. **Corto Plazo**
   - Aumentar el dataset
   - Implementar técnicas de normalización
   - Desarrollar pipeline de preprocesamiento

2. **Mediano Plazo**
   - Entrenar modelos iniciales
   - Evaluar diferentes arquitecturas
   - Optimizar hiperparámetros

3. **Largo Plazo**
   - Implementar sistema en tiempo real
   - Desarrollar interfaz de usuario
   - Realizar pruebas de campo

## 10. Referencias

[https://medium.com/@riddhisi238/real-time-pose-estimation-from-video-using-mediapipe-and-opencv-in-python-20f9f19c77a6]