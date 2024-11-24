import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import mediapipe as mp
import cv2
from collections import deque, Counter
import pickle
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

    # === Nuevas características para Caminando ===
    # Movimiento vertical alternado de rodillas
    features['knee_alternation'] = np.abs(
        features['y25'] - features['y26']  # Diferencia de altura entre rodillas
    )
    
    # Movimiento horizontal de pies
    features['feet_movement'] = np.abs(
        (features['x27'] - features['x28'])  # Distancia horizontal entre pies
    )
    
    # Balanceo de brazos
    features['arm_swing'] = np.abs(
        (features['x15'] - features['x16']) +  # Distancia entre muñecas
        (features['x13'] - features['x14'])    # Distancia entre codos
    )
    
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
        self.prediction_history = deque(maxlen=10)
        
    def get_stable_prediction(self):
        if not self.prediction_history:
            return None, 0.0
            
        recent_predictions = list(self.prediction_history)[-10:]
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
        knee_flex_threshold = 45.0
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
            print(f"✗ Torso no vertical: {torso_vertical:.1f}° > {torso_vertical_max}°")
            
        return confidence
    
    def verify_walking(self, features):
        confidence = 1.0
    
        # Umbrales para caminata
        feet_distance_min = 0.15     # Mínima separación de pies
        lateral_sway_min = 0.1      # Mínima oscilación lateral
        
        # Verificar distancia entre pies
        feet_dist = features['feet_distance'].values[0]
        if feet_dist < feet_distance_min:
            confidence -= 0.3
            print(f"✗ Poca separación de pies: {feet_dist:.2f} < {feet_distance_min}")
        
        # Verificar oscilación lateral
        lateral_sway = features['lateral_sway'].values[0]
        if lateral_sway < lateral_sway_min:
            confidence -= 0.3
            print(f"✗ Poca oscilación lateral: {lateral_sway:.2f} < {lateral_sway_min}")
        
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
            if standing_confidence > 0.6:
                confidence += 0.2
                print("✓ Verificación Parado: postura correcta (+0.2)")
            else:
                confidence -= 0.3
                print("✗ Verificación Parado: postura incorrecta (-0.3)")
                if standing_confidence < 0.4:  # Si la confianza es muy baja
                    stable_prediction = 'Sentado'
                    print("! Cambiando predicción a Sentado")
        # Agregar verificación para caminata
        elif stable_prediction in ['Caminando_Frente', 'Caminando_Espalda']:
            walking_confidence = self.verify_walking(features)
            if walking_confidence > 0.6:
                confidence += 0.2
                print("✓ Verificación Caminando: movimiento correcto (+0.2)")
            else:
                confidence -= 0.3
                print("✗ Verificación Caminando: poco movimiento (-0.3)")
                if walking_confidence < 0.4:
                    stable_prediction = 'Parado'
                    print("! Cambiando predicción a Parado")
                
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

def main():
    # Cargar el modelo pre-entrenado
    model_path = "../models/svm_model.pkl"
    scaler_path = "../models/scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Error: No se encontraron los archivos del modelo pre-entrenado")
        return
        
    print("Cargando modelo pre-entrenado...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Inicializar detector
    detector = PoseDetector(model, scaler)
    
    # Capturar video de la cámara web
    print("Iniciando captura de video...")
    cap = cv2.VideoCapture(0)  # 0 para la cámara web predeterminada
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar frame")
            break
        
        prediction, confidence, landmarks, features = detector.process_frame(frame)
        
        if landmarks:
            detector.draw_landmarks(frame, landmarks)
        
        if prediction:
            cv2.putText(frame, f"Accion: {prediction}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confianza: {confidence:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
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
        
        cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()