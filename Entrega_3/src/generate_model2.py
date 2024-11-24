import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from scipy.signal import savgol_filter
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

def main():
    # Crear directorio para modelos si no existe
    models_dir = "../models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    print("=== GENERACIÓN DEL MODELO ===")
    
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
    
    # 10. Guardar modelo y scaler
    print("\nGuardando modelo y scaler...")
    with open(f"{models_dir}/svm_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open(f"{models_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler_final, f)
    
    print(f"\nModelo y scaler guardados en {models_dir}/")

if __name__ == "__main__":
    main()