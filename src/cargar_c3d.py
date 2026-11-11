import os
import glob
import ezc3d
import pandas as pd
import numpy as np

# 1. Ruta de los datos (Asegúrate de que sea la correcta)
CARPETA_DATOS = r"DS_10283_2913/orchestral conducting mocap data/data share files"
ARCHIVO_SALIDA = "datos_crudos.parquet"

# 2. Marcadores de interés
marcadores_interes = [
    "LFHD", "RFHD", "LBHD", "RBHD",       # Cabeza
    "LSHO", "RSHO",                       # Hombros
    "LELB", "RELB",                       # Codos
    "LWRA", "LWRB", "RWRA", "RWRB",       # Muñecas
    "LFIN", "RFIN",                       # Dedos
    "Baton End", "Baton Shaft",           # Batuta
    "C7", "CLAV", "STRN", "T10"           # Torso
]

def procesar_archivo(ruta_c3d):
    c3d = ezc3d.c3d(ruta_c3d)
    nombres_todos = c3d['parameters']['POINT']['LABELS']['value']
    frecuencia = c3d['header']['points']['frame_rate']
    
    # ACCESO DIRECTO A LOS DATOS (Shape: 4 x N_marcadores x N_frames)
    # La primera dimensión tiene [x, y, z, residual]
    puntos = c3d['data']['points']
    num_frames = puntos.shape[2]
    
    datos_dict = {}
    
    for marcador in marcadores_interes:
        if marcador in nombres_todos:
            idx = nombres_todos.index(marcador)
            # Extraemos x, y, z y DIVIDIMOS ENTRE 1000 para pasar de mm a metros
            datos_dict[f"{marcador}_x"] = puntos[0, idx, :] / 1000.0
            datos_dict[f"{marcador}_y"] = puntos[1, idx, :] / 1000.0
            datos_dict[f"{marcador}_z"] = puntos[2, idx, :] / 1000.0
        else:
            print(f"  Advertencia: Marcador '{marcador}' no encontrado.")
    
    df = pd.DataFrame(datos_dict)
    df.insert(0, 'frame', np.arange(num_frames))
    df.insert(1, 'tiempo_seg', df['frame'] / frecuencia)
    
    nombre_archivo = os.path.basename(ruta_c3d).replace('.c3d', '')
    df.insert(0, 'recording_id', nombre_archivo)
    
    return df

def main():
    archivos = glob.glob(os.path.join(CARPETA_DATOS, "*.c3d"))
    print(f"Se encontraron {len(archivos)} archivos C3D.")
    
    lista_dataframes = []
    
    for i, archivo in enumerate(archivos):
        nombre = os.path.basename(archivo)
        print(f"Procesando [{i+1}/{len(archivos)}]: {nombre}...")
        
        df = procesar_archivo(archivo)
        lista_dataframes.append(df)
        
        if i == 0:
            # Guardamos con separador ';' y decimal ',' para Excel en español
            df.to_csv("muestra_datos.csv", index=False, sep=';', decimal=',')
            print(f"  -> Se guardó 'muestra_datos.csv' para inspección.")

    print("\nUniendo todos los datos...")
    df_final = pd.concat(lista_dataframes, ignore_index=True)
    
    print(f"Guardando datos consolidados en {ARCHIVO_SALIDA}...")
    df_final.to_parquet(ARCHIVO_SALIDA, index=False)
    
    print("\n¡Proceso terminado con éxito!")
    print(f"Shape del DataFrame final: {df_final.shape}")

if __name__ == "__main__":
    main()