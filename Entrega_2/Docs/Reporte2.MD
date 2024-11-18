# Reporte Entrega 2: Preprocesamiento y Modelo de Clasificación de Poses

## 1. Preprocesamiento de Datos

### 1.1 Visualización y Limpieza de Outliers
En esta etapa visualizamos la distribución de las coordenadas (x, y, z) de los landmarks y su relación con cada tipo de acción mediante boxplots. Esto nos permitió identificar patrones y posibles outliers en los datos antes de proceder con el entrenamiento del modelo.

Aplicamos una limpieza de outliers mediante el método IQR, conservando el 81.14% de los datos originales (9,168 de 11,299 registros), lo que indica que el proceso de limpieza eliminó una cantidad razonable de valores atípicos mientras mantiene la mayoría de los datos relevantes para el análisis.

### 1.2 Normalización
Realizamos la separación de características (X) y etiquetas (y), excluyendo las columnas 'action' y 'source_video'. Luego, aplicamos la normalización de datos utilizando StandardScaler, que estandariza las características a una media de 0 y desviación estándar de 1.

### 1.3 Suavizado de Señales
Aplicamos un suavizado de señales utilizando el filtro Savitzky-Golay, que ayuda a reducir el ruido en los datos mientras preserva las características importantes de las señales de movimiento. El filtro se aplicó con una ventana de 5 frames y un orden polinomial de 2.

### 1.4 Generación de Características
Generamos características adicionales para enriquecer nuestro conjunto de datos:
- Velocidades entre landmarks consecutivos
- Ángulos entre articulaciones clave (cabeza-hombros, hombros-caderas, codos y rodillas)
- Inclinaciones del tronco y laterales
- Distancias relativas entre puntos importantes (manos y pies)

## 2. Modelo de Clasificación

### 2.1 Implementación
Implementamos un modelo de clasificación SVM (Support Vector Machine) para reconocer los diferentes tipos de acciones. Para esto, primero preparamos los datos separando las características (X) y las etiquetas (y), y dividimos el conjunto en datos de entrenamiento (80%) y prueba (20%). Luego, entrenamos el modelo SVM con kernel lineal.

### 2.2 Resultados
Los resultados del modelo SVM muestran un rendimiento sobresaliente con un accuracy general del 92%. El modelo demuestra una precisión perfecta o casi perfecta (98-100%) en la clasificación de acciones dinámicas como caminar en diferentes direcciones y giros. Sin embargo, presenta un rendimiento ligeramente menor en acciones estáticas como "Parado" (85%) y "Sentado" (82%), lo que sugiere que estas posturas comparten características similares que pueden dificultar su distinción. A pesar de estas pequeñas variaciones, el modelo mantiene un f1-score promedio del 95%, indicando un equilibrio robusto entre precisión y recall en todas las clases.

## 3. Estructura del Proyecto

── src / EDA_Proyecto.ipynb

── db/poses_dataset.csv - poses_dataset_processed.csv

── models/svm_model.pkl

── docs/Reporte2.MD

## 4. Conclusiones
El preprocesamiento exhaustivo y la generación de características adicionales han permitido obtener un modelo robusto para la clasificación de poses y acciones. El modelo SVM muestra un excelente rendimiento general, aunque hay espacio para mejoras en la distinción de acciones estáticas.