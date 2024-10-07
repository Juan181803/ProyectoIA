# Proyecto: Sistema de Detección de Objetos y Reconocimiento de Comportamiento en Videos de Vigilancia

Este proyecto implica desarrollar un sistema que no solo detecte objetos en escenas de vigilancia (personas, carros, etc.), sino que también pueda analizar patrones de comportamiento para identificar acciones como caminar, correr, sentarse o comportamientos sospechosos. Esto nos permitirá tener un sistema de vigilancia mucho más avanzado y robusto, pues se espera que con la detección de estos comportamientos sospechosos se pueda hacer seguimiento o vigilancia de una persona.

Integrantes :
 - Juan Felipe Ramirez A00382637
 - Collin Gonzalez 


## 1. Pregunta de investigación
**Pregunta principal**: ¿Cómo podemos desarrollar un sistema eficiente en tiempo real para detectar múltiples objetos y reconocer comportamientos humanos en videos de vigilancia mediante técnicas de visión por computadora y aprendizaje automático?

**Subpreguntas**:
- ¿Qué características visuales y patrones de movimiento son clave para reconocer comportamientos en videos?
- ¿Cómo se puede lograr un balance entre la precisión del modelo de detección y la velocidad de procesamiento en tiempo real?
- ¿Cómo puede el sistema identificar comportamientos sospechosos (por ejemplo, una persona merodeando) en diferentes escenarios?

## 2. Tipo de problema
Este problema combina:
- **Detección de objetos** (visión por computadora): Identificación de múltiples objetos en escenas de video, como personas, vehículos o bicicletas.
- **Reconocimiento de acciones**: Detección y clasificación de actividades humanas como caminar, correr, detenerse, etc.
- **Análisis en tiempo real**: Procesar video en tiempo real para monitorear el entorno de manera continua.

## 3. Metodología: Adaptación de CRISP-DM
**Comprensión del problema**:
- **Contexto**: Los sistemas de vigilancia actuales requieren la automatización de la detección de objetos y el análisis de comportamientos para aumentar la seguridad en áreas públicas y privadas.
- **Objetivos**: Desarrollar un sistema capaz de:
  - Detectar y clasificar objetos en una escena (por ejemplo, personas, autos).
  - Reconocer comportamientos o patrones de movimiento (caminar, correr, comportamiento sospechoso).
  - Funcionar en tiempo real para aplicaciones de vigilancia.

**Comprensión de los datos**:
- **Plan de recopilación de datos**: Utilizar conjuntos de datos públicos de videos de vigilancia, como el DukeMTMC, o capturar videos de vigilancia simulados en entornos controlados.
- **Requisitos iniciales de datos**:
  - Conjunto de videos con múltiples objetos y acciones etiquetadas (correr, caminar, merodear, etc.).
  - Videos con diferentes perspectivas de cámara y variaciones en el número de objetos presentes.

**Preparación de datos**:
- **Preprocesamiento**:
  - Filtrado de ruido en los videos.
  - Segmentación de las escenas de interés donde ocurren las acciones clave.
  - Anotación de las actividades y los objetos en los videos.

**Extracción de características**:
- **Detección de objetos**: Usar modelos preentrenados como YOLO o SSD para detectar y clasificar objetos en tiempo real.
- **Reconocimiento de acciones**: Usar técnicas de análisis de series temporales para rastrear y analizar los movimientos de los objetos detectados.

## 4. Métricas de rendimiento
- **Precisión de la detección de objetos**: Mide cuántos objetos son detectados correctamente en comparación con los etiquetados manualmente.
- **Recall de detección de comportamientos**: Mide la capacidad del sistema para identificar correctamente comportamientos como caminar, correr o merodear.
- **FPS (Fotogramas por segundo)**: Evalúa el rendimiento en tiempo real del sistema.
- **Exactitud de clasificación de acciones**: Mide la precisión en la clasificación de actividades humanas.

## 5. Consideraciones éticas
- **Privacidad y consentimiento**: El sistema debe estar alineado con las normativas locales sobre el uso de cámaras de vigilancia y protección de datos.
- **Transparencia**: Explicar las limitaciones del sistema y las posibles situaciones en las que podría fallar (por ejemplo, en escenarios con baja iluminación o oclusión de objetos).
- **Mitigación de sesgos**: Asegurarse de que el sistema funcione de manera justa en diferentes entornos y tipos de población.

## 6. Próximos pasos
- **Semana 11-12**:
  - Obtener un conjunto de videos de vigilancia y empezar el preprocesamiento.
  - Implementar la detección de objetos usando un modelo preentrenado como YOLO.
- **Semana 13**:
  - Integrar el reconocimiento de comportamientos usando análisis de movimiento.
  - Evaluar el rendimiento en videos de prueba y ajustar el sistema.
- **Semana 14**:
  - Mejorar la precisión del sistema optimizando el modelo.
  - Realizar pruebas en escenarios simulados en tiempo real.

## 7. Estrategias adicionales de adquisición de datos
- **Uso de datos públicos**: Aprovechar conjuntos de datos de videos de vigilancia como el DukeMTMC o UCF Crime para entrenar y evaluar el sistema.
- **Aumento de datos**: Generar nuevos escenarios alterando los videos existentes (cambiando las perspectivas o velocidades de los objetos).
- **Simulación de comportamientos**: Usar entornos controlados para grabar videos adicionales que simulen comportamientos sospechosos o inusuales.
