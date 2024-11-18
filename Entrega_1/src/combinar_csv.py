import pandas as pd
import os
from pathlib import Path

def combine_csv_files(input_dir, output_dir):
    # Crear directorio de salida si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Lista para almacenar todos los DataFrames
    all_dfs = []
    
    # Leer todos los archivos CSV
    for csv_file in os.listdir(input_dir):
        if csv_file.endswith('.csv'):
            print(f"Procesando {csv_file}...")
            file_path = os.path.join(input_dir, csv_file)
            
            # Leer CSV y agregar columna con el nombre del archivo original
            df = pd.read_csv(file_path)
            df['source_video'] = csv_file.replace('.csv', '')
            
            all_dfs.append(df)
    
    # Combinar todos los DataFrames
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Guardar el DataFrame combinado
        output_path = os.path.join(output_dir, 'poses_dataset.csv')
        combined_df.to_csv(output_path, index=False)
        
        print(f"\nDataset combinado creado exitosamente en: {output_path}")
        print(f"Dimensiones del dataset: {combined_df.shape}")
        print(f"\nDistribución de acciones:")
        print(combined_df['action'].value_counts())
    else:
        print("No se encontraron archivos CSV para procesar")

if __name__ == "__main__":
    input_dir = "../datos_procesados"
    output_dir = "../../db"  # Sube un nivel y entra a la carpeta db
    combine_csv_files(input_dir, output_dir)