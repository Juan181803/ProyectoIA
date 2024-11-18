import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from scipy.signal import savgol_filter
import mediapipe as mp
import cv2
from collections import deque

# 6. Generación de características
def generate_features(df):
    features = df.copy()
        
    # Velocidades entre landmarks consecutivos
    print("Calculando velocidades...")
    for i in range(0, 33):
        if i + 1 < 33:
            features[f'vel_{i}_{i+1}'] = np.sqrt(
                (features[f'x{i+1}'] - features[f'x{i}']).pow(2) +
                (features[f'y{i+1}'] - features[f'y{i}']).pow(2) +
                (features[f'z{i+1}'] - features[f'z{i}']).pow(2)
            )
        
    # Ángulos entre articulaciones
    print("Calculando ángulos...")
    features['angle_head_shoulders'] = np.degrees(
        np.arctan2(features['y11'] - features['y12'],
                  features['x11'] - features['x12'])
    )
    features['angle_shoulders_hips'] = np.degrees(
            np.arctan2(features['y23'] - features['y24'],
                      features['x23'] - features['x24'])
    )
    features['angle_right_elbow'] = np.degrees(
            np.arctan2(features['y15'] - features['y13'],
                      features['x15'] - features['x13'])
    )
    features['angle_left_elbow'] = np.degrees(
            np.arctan2(features['y16'] - features['y14'],
                      features['x16'] - features['x14'])
    )
    features['angle_right_knee'] = np.degrees(
        np.arctan2(features['y25'] - features['y23'],
                  features['x25'] - features['x23'])
    )
    features['angle_left_knee'] = np.degrees(
            np.arctan2(features['y26'] - features['y24'],
                      features['x26'] - features['x24'])
    )
        
    # Inclinaciones
    print("Calculando inclinaciones...")
    features['trunk_inclination'] = np.degrees(
            np.arctan2(
                (features['y11'] + features['y12'])/2 - (features['y23'] + features['y24'])/2,
                (features['x11'] + features['x12'])/2 - (features['x23'] + features['x24'])/2
            )
    )
    features['lateral_inclination'] = np.degrees(
        np.arctan2(features['y11'] - features['y12'],
                  features['x11'] - features['x12'])
    )
        
    # Distancias relativas
    print("Calculando distancias relativas...")
    features['hands_distance'] = np.sqrt(
        (features['x19'] - features['x20']).pow(2) +
        (features['y19'] - features['y20']).pow(2) +
        (features['z19'] - features['z20']).pow(2)
        )
    features['feet_distance'] = np.sqrt(
        (features['x31'] - features['x32']).pow(2) +
        (features['y31'] - features['y32']).pow(2) +
        (features['z31'] - features['z32']).pow(2)
        )
        
    return features

class PoseDetector:
    def __init__(self, model, scaler):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.model = model
        self.scaler = scaler
        self.landmark_buffer = deque(maxlen=5)  # Para suavizado temporal
        
    def process_frame(self, frame):
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar frame
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Extraer landmarks
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Crear DataFrame con los landmarks
            columns = []
            for i in range(33):  # 33 landmarks
                columns.extend([f'x{i}', f'y{i}', f'z{i}'])
            df = pd.DataFrame([landmarks], columns=columns)
            
            # Agregar al buffer para suavizado
            self.landmark_buffer.append(df)

            if len(self.landmark_buffer) == 5:  # Buffer lleno
                # Aplicar suavizado
                df_smoothed = pd.concat(list(self.landmark_buffer)).mean()
                df_smoothed = pd.DataFrame([df_smoothed.values], columns=df_smoothed.index)
                
                # Generar características
                features = generate_features(df_smoothed)
                
                # Escalar características
                features_scaled = self.scaler.transform(features)
                
                # Predecir
                prediction = self.model.predict(features_scaled)[0]
                return prediction, results.pose_landmarks
            
        return None, results.pose_landmarks

def main():
    print("=== VALIDACIÓN DEL MODELO ===")
    
    # 1. Cargar datos
    print("\nCargando datos...")
    data_path = "../../db/poses_dataset.csv"
    df = pd.read_csv(data_path)
    
    # 2. Limpieza inicial y detección de outliers
    print("\nLimpiando datos...")
    df_cleaned = df.copy()
    
    # Filtrar outliers usando IQR para cada coordenada por tipo de acción
    for action in df['action'].unique():
        action_mask = df_cleaned['action'] == action
        
        for column in df_cleaned.columns:
            if column != 'action' and column != 'source_video':
                Q1 = df_cleaned.loc[action_mask, column].quantile(0.25)
                Q3 = df_cleaned.loc[action_mask, column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df_cleaned.loc[action_mask, column] >= lower_bound) & \
                              (df_cleaned.loc[action_mask, column] <= upper_bound)
                
                df_cleaned.loc[action_mask & ~outlier_mask, column] = np.nan
    
    # Eliminar filas con valores NaN
    df_cleaned = df_cleaned.dropna()
    
    print(f"Registros originales: {len(df)}")
    print(f"Registros después de limpieza: {len(df_cleaned)}")
    print(f"Porcentaje de datos conservados: {(len(df_cleaned)/len(df))*100:.2f}%")
    
    # 3. Separar features y target
    X = df_cleaned.drop(['action', 'source_video'], axis=1)
    y = df_cleaned['action']
    
    # 4. Normalización inicial
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 5. Suavizado de señales
    def smooth_signals(df, window_length=5, polyorder=2):
        df_smoothed = df.copy()
        for coord in ['x', 'y', 'z']:
            cols = [col for col in df.columns if col.startswith(coord)]
            for col in cols:
                df_smoothed[col] = savgol_filter(df[col], window_length, polyorder)
        return df_smoothed
    
    X_smoothed = smooth_signals(X_scaled)
    
    X_with_features = generate_features(X_smoothed)
    
    # 7. Preparación final de datos
    print("\nPreparando datos finales...")
    X = X_with_features
    print(f'Shape de X: {X.shape}')
    print(f'Shape de y: {y.shape}')
    
    # 8. Split de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f'\nShape de X_train: {X_train.shape}')
    print(f'Shape de X_test: {X_test.shape}')
    
    # 9. Escalado final
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    X_test_scaled = scaler_final.transform(X_test)
    
    # 10. Entrenamiento y evaluación
    print("\nEntrenando modelo SVM...")
    model = SVC(kernel='rbf', C=10, gamma='scale')
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    print(f'\nPrimeras predicciones: {y_pred[:5]}')
    print(f'Etiquetas reales: {y_test[:5].values}')
    
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

    # Después del entrenamiento, inicializar detector
    detector = PoseDetector(model, scaler_final)
    
    # Abrir video
    video_path = "../../Entrega_1/videos/Sentado-12.mp4"  # Ajusta la ruta según necesites
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Procesar frame
        prediction, landmarks = detector.process_frame(frame)
        
        if prediction:
            # Dibujar predicción en el frame
            cv2.putText(frame, f"Accion: {prediction}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mostrar frame
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()