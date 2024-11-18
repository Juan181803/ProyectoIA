import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np
from pathlib import Path

class PoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_landmarks(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames_data = []
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            # Convertir BGR a RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            
            if results.pose_landmarks:
                frame_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                frames_data.append(frame_landmarks)
                
        cap.release()
        return frames_data
    
def get_action_label(filename):
    if "CaminandoE" in filename:
        return "Caminando_Espalda"
    elif "CaminandoF" in filename:
        return "Caminando_Frente"
    elif "Giro" in filename:
        return "Giro"
    elif "Parado" in filename:
        return "Parado"
    elif "Sentado" in filename:
        return "Sentado"
    elif "Quieto" in filename:
        return "Quieto"
    elif "CaminandoL" in filename:
        return "Caminando_Lado"
    else:
        return "Desconocido"

def process_videos(videos_dir, output_dir):
    extractor = PoseExtractor()
    
    # Crear directorio de salida si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for video_file in os.listdir(videos_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            print(f"Procesando {video_file}...")
            
            # Obtener etiqueta de la acción
            action_label = get_action_label(video_file)
            
            # Procesar video
            video_path = os.path.join(videos_dir, video_file)
            landmarks_data = extractor.extract_landmarks(video_path)
            
            # Crear DataFrame
            columns = []
            for i in range(33):  # MediaPipe detecta 33 puntos
                columns.extend([f'x{i}', f'y{i}', f'z{i}'])
            
            df = pd.DataFrame(landmarks_data, columns=columns)
            df['action'] = action_label
            
            # Guardar CSV
            output_path = os.path.join(output_dir, f"{video_file.split('.')[0]}.csv")
            df.to_csv(output_path, index=False)

if __name__ == "__main__":
    videos_dir = "../videos"
    output_dir = "../datos_procesados"
    process_videos(videos_dir, output_dir)