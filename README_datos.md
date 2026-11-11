# Documentación de Datos Crudos - Proyecto de Dirección Orquestal

Este documento describe la estructura y extracción de los datos de captura de movimiento (mocap) del dataset de la Universidad de Edimburgo.

## 1. Archivos Generados
*   **`cargar_c3d.py`**: Script de Python que procesa los archivos C3D originales.
*   **`datos_crudos.parquet`**: Archivo final consolidado (recomendado para usar en el modelado por su velocidad y ligereza).
*   **`muestra_datos.csv`**: Muestra del primer archivo en formato CSV (separado por `;` y con coma decimal) para inspección visual.

## 2. Origen de los Datos
*   **Total de archivos procesados:** 62 archivos `.c3d`.
*   **Frecuencia de muestreo:** 120 Hz (120 frames por segundo).
*   **Duración total:** Aproximadamente 976,408 frames en total (aprox. 2.2 horas de grabación).

## 3. Estructura del Dataset
El dataset consolidado tiene **976,408 filas** y **63 columnas**:

*   **Metadatos:**
    *   `recording_id`: Identificador único de la grabación (ej. `Bartok_Conductor01_trial01`).
    *   `frame`: Número de frame original (0 a N).
    *   `tiempo_seg`: Tiempo en segundos desde el inicio de la grabación (`frame / 120`).
*   **Coordenadas (60 columnas):**
    *   Para cada marcador anatómico, hay 3 columnas: `_x`, `_y`, `_z`.
    *   **Unidades:** Metros (se aplicó una división entre 1000 a los datos originales, que venían en milímetros).
    *   **Marcadores extraídos:** 
        *   *Cabeza:* `LFHD`, `RFHD`, `LBHD`, `RBHD`
        *   *Hombros:* `LSHO`, `RSHO`
        *   *Codos:* `LELB`, `RELB`
        *   *Muñecas:* `LWRA`, `LWRB`, `RWRA`, `RWRB`
        *   *Dedos:* `LFIN`, `RFIN`
        *   *Batuta:* `Baton End`, `Baton Shaft`
        *   *Torso:* `C7`, `CLAV`, `STRN`, `T10`

## 4. Anomalías Detectadas (¡Importante para la Persona 2!)
Durante el procesamiento se detectaron las siguientes ausencias de marcadores en archivos específicos:
*   **Directores sin batuta:** Los archivos de `Conductor05` (en Bartok, Dvorak y Mozart) no contienen los marcadores `Baton End` ni `Baton Shaft`. El algoritmo de clustering deberá manejar esto (quizás ignorando esas columnas para ese director o usando imputación).
*   **Marcadores faltantes en tomas específicas:**
    *   `Bartok_Conductor06_trial03_1`: Faltan `LWRA`, `LWRB`, `LFIN` (mano izquierda).
    *   `Dvorak_Conductor06_trial03_2`: Falta `STRN` (esternón).

**Nota sobre valores faltantes (Oclusiones):** En la captura de movimiento es normal que algunos frames tengan valores en `0.0` si el sistema perdió el marcador por un instante. La Persona 2 deberá decidir si aplica interpolación o suavizado.