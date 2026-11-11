# Pipeline mejorado (v2.0)

Ejecutar **desde la raíz del proyecto** (las rutas son relativas: `data/`, `work/`, `outputs/`, `demo/`). Crear antes `work/` y `outputs/figuras/`.

| Orden | Script | Qué hace | Salida |
|---|---|---|---|
| 1 | `load_raw.py` | Lee `data/datos_crudos.parquet` (con `pqread.py`/`thrift_pq.py`, un lector mínimo que no requiere PyArrow; si tienes PyArrow puedes cargar con pandas) | `work/raw.npy`, `rid.npy`, `cols.json` |
| 2 | `mocap_pipeline.py` | Fusiona variantes `_1/_2`, limpia y calcula variables | `work/features_all.pkl`, `clean_info.json`, `fusion.json` |
| 3 | `modelado_a.py` | Métricas para k=2..8 (K-Means, Ward, GMM; conjuntos BASE/MEJ/VENT) | `work/metricas_k.pkl`, `gmm.pkl`, `features_sel.pkl` |
| 4 | `modelado_b.py` | Modelos finales, estabilidad, PCA, t-SNE, NMI | `work/D_labels.pkl`, `estabilidad.pkl`, `nmi.pkl` |
| 5 | `analisis_c.py` | Episodios, transiciones, estadísticas por toma, batuta, sensibilidad | `work/res_c.json`, `sens.pkl`, `tomas.pkl` |
| 6 | `figuras1.py`, `figuras2.py`, `figuras3.py` | Las 32 figuras (necesitan además `work/names.json` con los nombres de clusters, `work/pca_var.json`, `work/energy_corr.json`, `work/sil_k5.json`, que generan `figuras2/3` y pasos intermedios del análisis) | `outputs/figuras/*.png` |
| 7 | `exportar_artefactos_dashboard.py` | Exporta los CSV/JSON compactos del dashboard | `demo/data/*` |

Notas: el análisis se ejecutó en un equipo de un solo núcleo con 3 GB de RAM (≈ 10 minutos en total). Semillas fijas (42 en los modelos).
