import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from scipy.signal import savgol_filter
import mediapipe as mp
import cv2
from collections import deque, Counter
import os

def generate_features(df):
    features = df.copy()
    
    # === Características para Sentado ===
    # Distancia vertical entre cadera y rodilla
    features['hip_knee_distance'] = np.sqrt(
        (features['y23'] - features['y25']).pow(2) +  # Cadera derecha a rodilla derecha
        (features['y24'] - features['y26']).pow(2)    # Cadera izquierda a rodilla izquierda
    )
    
    # Altura relativa de caderas
    features['hip_height'] = (features['y23'] + features['y24'])/2
    
    # Ángulos de flexión de rodillas
    features['right_knee_flex'] = np.degrees(
        np.arctan2(features['y27'] - features['y25'],  # Tobillo - rodilla
                  features['y25'] - features['y23'])   # Rodilla - cadera
    )
    features['left_knee_flex'] = np.degrees(
        np.arctan2(features['y28'] - features['y26'],
                  features['y26'] - features['y24'])
    )
    
    # === Características para Caminando ===
    # Distancia entre pies
    features['feet_distance'] = np.sqrt(
        (features['x27'] - features['x28']).pow(2) +
        (features['y27'] - features['y28']).pow(2)
    )
    
    # Movimiento vertical del centro de masa
    features['com_height'] = (features['y23'] + features['y24'])/2
    
    # Oscilación lateral
    features['lateral_sway'] = np.abs(features['x23'] - features['x24'])
    
    # === Características para Parado ===
    # Verticalidad del torso
    features['torso_vertical'] = np.degrees(
        np.arctan2(
            (features['y11'] + features['y12'])/2 - (features['y23'] + features['y24'])/2,
            (features['x11'] + features['x12'])/2 - (features['x23'] + features['x24'])/2
        )
    )
    
    # Rotación de hombros
    features['shoulder_rotation'] = np.degrees(
        np.arctan2(
            features['y12'] - features['y11'],
            features['x12'] - features['x11']
        )
    )
    
    return features

class PoseDetector:
    def __init__(self, model, scaler):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.model = model
        self.scaler = scaler
        self.prediction_history = deque(maxlen=10)  # Usar deque con tamaño máximo
        
    def get_stable_prediction(self):
        if not self.prediction_history:
            return None, 0.0
            
        recent_predictions = list(self.prediction_history)[-10:]  # Convertir a lista
        prediction_counts = Counter(recent_predictions)
        most_common = prediction_counts.most_common(1)[0]
        
        # Calcular confianza base
        confidence = 1.0
        
        # Verificar estabilidad de predicción
        if len(recent_predictions) >= 5:
            last_5 = recent_predictions[-5:]
            if len(set(last_5)) == 1:  # Si las últimas 5 predicciones son iguales
                confidence += 0.2
                
        return most_common[0], confidence
        
    def verify_standing(self, features):
        knee_flex_threshold = 45.0  # Ajustado para ser más estricto
        hip_height_min = 0.45
        feet_distance_max = 0.3
        torso_vertical_max = 10.0
        
        confidence = 1.0
        
        # Verificar flexión de rodilla
        knee_flex = abs(features['right_knee_flex'].values[0])
        if knee_flex > knee_flex_threshold:
            confidence -= 0.4
            print(f"✗ Flexión de rodilla excesiva: {knee_flex:.1f}° > {knee_flex_threshold}°")
            
        # Verificar altura de cadera
        hip_height = features['hip_height'].values[0]
        if hip_height < hip_height_min:
            confidence -= 0.2
            print(f"✗ Altura de cadera baja: {hip_height:.2f} < {hip_height_min}")
            
        # Verificar distancia entre pies
        feet_distance = features['feet_distance'].values[0]
        if feet_distance > feet_distance_max:
            confidence -= 0.1
            print(f"✗ Pies muy separados: {feet_distance:.2f} > {feet_distance_max}")
            
        # Verificar verticalidad del torso
        torso_vertical = abs(features['torso_vertical'].values[0])
        if torso_vertical > torso_vertical_max:
            confidence -= 0.2
            print(f"✗ Torso inclinado: {torso_vertical:.1f}° > {torso_vertical_max}°")
            
        return confidence

    def process_frame(self, frame):
        # Convertir BGR a RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        if not results.pose_landmarks:
            return None, 0.0, None, None
            
        # Extraer landmarks
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
            
        # Crear DataFrame con landmarks
        columns = []
        for i in range(33):
            columns.extend([f'x{i}', f'y{i}', f'z{i}'])
        features = pd.DataFrame([landmarks], columns=columns)
        
        # Generar características adicionales
        features = generate_features(features)
        
        # Escalar características
        features_scaled = self.scaler.transform(features)
        
        # Predecir
        prediction = self.model.predict(features_scaled)[0]
        self.prediction_history.append(prediction)
        
        # Obtener predicción estable
        stable_prediction, confidence = self.get_stable_prediction()
        
        # Verificar confianza adicional según la postura
        if stable_prediction == 'Parado':
            standing_confidence = self.verify_standing(features)
            if standing_confidence > 0.6:  # Solo si la verificación es positiva
                confidence += 0.2
                print("✓ Verificación Parado: postura correcta (+0.2)")
            else:
                confidence -= 0.3
                print("✗ Verificación Parado: postura incorrecta (-0.3)")
                stable_prediction = 'Sentado'  # Cambiar predicción si la verificación falla
                
        # Debug info
        print("\n=== ANÁLISIS DE PREDICCIONES ===")
        print(f"Últimas predicciones: {list(self.prediction_history)}")
        print(f"Conteo de predicciones: {dict(Counter(self.prediction_history))}")
        print(f"Predicción más común: {stable_prediction}")
        print(f"Confianza base: {confidence:.2f}")
        
        if features is not None:
            print("\nCaracterísticas detectadas:")
            print(f"- Flexión rodilla: {abs(features['right_knee_flex'].values[0]):.1f}°")
            print(f"- Altura cadera: {features['hip_height'].values[0]:.2f}")
            print(f"- Distancia pies: {features['feet_distance'].values[0]:.2f}")
            print(f"- Verticalidad torso: {features['torso_vertical'].values[0]:.1f}°")
            print(f"- Rotación hombros: {features['shoulder_rotation'].values[0]:.1f}°")
            print(f"- Oscilación lateral: {features['lateral_sway'].values[0]:.2f}")
        
        print(f"\nConfianza final: {confidence:.2f}")
        print("============================")
        
        return stable_prediction, confidence, results.pose_landmarks, features

    def draw_landmarks(self, frame, landmarks):
        self.mp_draw.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )

def list_available_videos():
    video_dir = "../../Entrega_1/videos"
    videos = []
    
    # Listar todos los archivos .mp4 en el directorio
    try:
        for i, file in enumerate(sorted(os.listdir(video_dir)), 1):
            if file.endswith('.mp4'):
                videos.append((i, file))
                print(f"{i}. {file}")
    except FileNotFoundError:
        print("Error: No se encontró el directorio de videos")
        return None
        
    if not videos:
        print("No se encontraron videos .mp4 en el directorio")
        return None
        
    return videos, video_dir

def select_video():
    print("\nVideos disponibles:")
    result = list_available_videos()
    if result is None:
        return None
        
    videos, video_dir = result
    
    while True:
        try:
            selection = int(input("\nSeleccione el número del video a analizar (0 para salir): "))
            if selection == 0:
                return None
            if 1 <= selection <= len(videos):
                selected_video = videos[selection-1][1]
                return os.path.join(video_dir, selected_video)
            else:
                print(f"Por favor seleccione un número entre 1 y {len(videos)}")
        except ValueError:
            print("Por favor ingrese un número válido")

def main():
    print("=== VALIDACIÓN DEL MODELO ===")
    
    # 1. Cargar datos
    print("\nCargando datos...")
    data_path = "../../db/poses_dataset.csv"
    df = pd.read_csv(data_path)
    
    # 2. Limpieza inicial
    print("\nLimpiando datos...")
    df_cleaned = df.copy()
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
    
    df_cleaned = df_cleaned.dropna()
    print(f"Registros originales: {len(df)}")
    print(f"Registros después de limpieza: {len(df_cleaned)}")
    print(f"Porcentaje de datos conservados: {(len(df_cleaned)/len(df))*100:.2f}%")
    
    # 3. Preparar datos
    X = df_cleaned.drop(['action', 'source_video'], axis=1)
    y = df_cleaned['action']
    
    # 4. Normalización inicial
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 5. Suavizado
    def smooth_signals(df, window_length=5, polyorder=2):
        df_smoothed = df.copy()
        for coord in ['x', 'y', 'z']:
            cols = [col for col in df.columns if col.startswith(coord)]
            for col in cols:
                df_smoothed[col] = savgol_filter(df[col], window_length, polyorder)
        return df_smoothed
    
    X_smoothed = smooth_signals(X_scaled)
    
    # 6. Generar características
    print("\nGenerando características...")
    X_with_features = generate_features(X_smoothed)
    print(f"\nShape de X: {X_with_features.shape}")
    print(f"Shape de y: {y.shape}")
    
    # 7. Split y escalado final
    X_train, X_test, y_train, y_test = train_test_split(
        X_with_features, y, test_size=0.2, random_state=42
    )
    print(f"\nShape de X_train: {X_train.shape}")
    print(f"Shape de X_test: {X_test.shape}")
    
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    X_test_scaled = scaler_final.transform(X_test)
    
    # 8. Entrenar modelo
    print("\nEntrenando modelo SVM...")
    model = SVC(kernel='rbf', C=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 9. Evaluar modelo
    y_pred = model.predict(X_test_scaled)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # 10. Detección en video
    video_path = select_video()
    if video_path is None:
        print("Programa terminado")
        return
        
    print(f"\nAnalizando video: {os.path.basename(video_path)}")
    detector = PoseDetector(model, scaler_final)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        prediction, confidence, landmarks, features = detector.process_frame(frame)
        
        if landmarks:
            detector.draw_landmarks(frame, landmarks)
        
        if prediction:
            cv2.putText(frame, f"Accion: {prediction}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confianza: {confidence:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Mostrar características relevantes para debug
            if features is not None:
                knee_flex = abs(features['right_knee_flex'].values[0])
                hip_height = features['hip_height'].values[0]
                torso_vertical = features['torso_vertical'].values[0]
                
                cv2.putText(frame, f"Knee Flex: {knee_flex:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Hip Height: {hip_height:.2f}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Torso Vertical: {torso_vertical:.1f}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()