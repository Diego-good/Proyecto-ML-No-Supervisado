# Dirección orquestal · Aprendizaje no supervisado (DAT-255) — v2.0

Pipeline de limpieza, variables biomecánicas y clustering sobre captura de movimiento (C3D) de directores de orquesta
(Universidad de Edimburgo), más un dashboard Streamlit con el análisis completo, el informe técnico y las 32 figuras
con su interpretación.

## Ejecutar el dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

No necesita el Parquet de datos crudos: usa los artefactos compactos de `demo/data/` (≈ 4 MB) y las figuras de
`demo/figuras_informe/`. El despliegue en Render sigue igual (`render.yaml`, `Procfile`).

## Estructura

| Ruta | Contenido |
|---|---|
| `app.py` | Dashboard (10 secciones) |
| `contenido.py` | Interpretación de cada figura (qué muestra, cómo leerla, conclusión) |
| `demo/data/` | Métricas, perfiles, ocupación por director/obra, series a 30 Hz, muestra de fotogramas con PCA/t-SNE, reporte de limpieza |
| `demo/figuras_informe/` | 32 figuras del informe |
| `demo/legacy/` | Demo v1.0 (calculada sobre una sola grabación); se conserva solo como referencia |
| `informe/` | Informe técnico (Word, 60+ páginas) |
| `src/` | Scripts originales (`cargar_c3d.py`, `features_clustering.py`) |
| `src/pipeline_mejorado/` | Pipeline v2.0 completo (ver su README) |
| `data/` | (no incluido) `datos_crudos.parquet` con las 62 grabaciones |

## Qué cambió respecto de la v1.0

- Se procesan las 62 grabaciones, que resultan ser **54 tomas únicas** (8 pares `_1/_2` con coordenadas idénticas y datos complementarios, fusionados).
- Los huecos de oclusión **largos (> 12 frames) ya no se rellenan por vecino**: antes congelaban la posición y daban velocidad cero artificial.
- Posiciones en un **marco corporal** (lateral, frontal, vertical respecto del torso): con posiciones absolutas los clusters identificaban al director (NMI ≈ 0,52 frente a ≈ 0,21).
- 26 variables (se descarta la energía cinética por ser redundante con la velocidad; la batuta se analiza aparte porque falta en el director 5).
- **K-Means con k=5** elegido por estabilidad (ARI 0,71 por mitades de tomas) y Davies-Bouldin (1,79); Silhouette 0,144 (estructura débil: el gesto se parece más a un continuo que a estados discretos).
- El dashboard v1.0 mostraba valores fijos en el código (k=3, Silhouette 0,231, Davies-Bouldin 1,70) que no coincidían con `demo/legacy/metricas_clustering.csv` (ganador k=2, Silhouette 0,229). La v2.0 lee todo de los artefactos.

## Alcance y límites

Los clusters son **regímenes cinemáticos descriptivos**. No se han validado contra etiquetas musicales, por lo que no se
afirma que correspondan a legato/staccato u otros conceptos musicales. Detalles en la sección «Metodología & Gobernanza» y en el informe.

## Datos

El conjunto original (62 archivos C3D) está en el repositorio DataShare de la Universidad de Edimburgo (handle 10283/2913).
Ver `README_datos.md`.
