# -*- coding: utf-8 -*-
"""Dashboard · Dirección Orquestal (DAT-255 Aprendizaje no supervisado) — versión 2.0
Lee artefactos compactos de demo/data y las figuras del informe en demo/figuras_informe."""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

import plotly.express as px
import plotly.graph_objects as go

from contenido import FIGURAS, SECCIONES_FIGURAS, DESCRIPCION_CLUSTERS, HALLAZGOS

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "demo", "data")
FIGDIR = os.path.join(BASE, "demo", "figuras_informe")
INFORME = os.path.join(BASE, "informe", "Informe_Aprendizaje_No_Supervisado_Mejorado.docx")
LOG_FILE = os.path.join(BASE, "bitacora_mlops.jsonl")

NOMBRES = ["C1 Compacto", "C2 Frontal moderado", "C3 Amplio", "C4 Asimétrico", "C5 Intenso"]
COLORES = {"C1 Compacto": "#3B82F6", "C2 Frontal moderado": "#F59E0B", "C3 Amplio": "#10B981",
           "C4 Asimétrico": "#EF4444", "C5 Intenso": "#8B5CF6"}
COLOR_DIR = ["#3B82F6", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#EC4899"]


# --------------------------------------------------------------------------- bitácora
def registrar_evento(accion, detalles):
    """Guarda un evento en el archivo JSONL (bitácora MLOps). Nunca interrumpe la app."""
    try:
        registro = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "accion": accion, "detalles": detalles}
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------- configuración y estilos
st.set_page_config(page_title="Dashboard Dirección Orquestal | MLOps", page_icon="🎼", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    [data-testid="stSidebar"] { background-color: #111827 !important; border-right: 1px solid #1F2937;
        min-width: 300px !important; max-width: 300px !important; }
    [data-testid="collapsedControl"] { display: none; }
    h1, h2, h3, h4 { font-family: 'Inter', sans-serif !important; color: #F9FAFB !important; }
    p, li, div.stMarkdown { font-family: 'Inter', sans-serif; color: #D1D5DB; line-height: 1.7; font-size: 0.95rem; }
    .metric-box { background-color: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 18px;
        border-left: 4px solid #3B82F6; margin-bottom: 15px; transition: transform 0.2s ease; }
    .metric-box:hover { transform: translateY(-3px); }
    .metric-box.green { border-left-color: #10B981; } .metric-box.orange { border-left-color: #F59E0B; }
    .metric-box.purple { border-left-color: #8B5CF6; } .metric-box.red { border-left-color: #EF4444; }
    .metric-label { font-size: 0.72rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    .metric-value { font-family: 'Courier New', monospace; font-size: 2rem; color: #F9FAFB; font-weight: bold; margin-top: 5px; }
    .metric-sub { font-size: 0.78rem; color: #6B7280; margin-top: 5px; }
    .top-header { background-color: #111827; padding: 20px 30px; border-radius: 10px; border: 1px solid #1F2937;
        display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .header-title { font-size: 1.8rem; font-weight: 700; margin: 0; color: #F9FAFB; letter-spacing: -0.5px; }
    .header-subtitle { font-size: 0.95rem; color: #9CA3AF; margin: 5px 0 0 0; }
    .status-badge { background-color: rgba(16,185,129,0.1); color: #10B981; padding: 8px 16px; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600; border: 1px solid #10B981; }
    .analysis-block { background-color: #1F2937; border-radius: 8px; padding: 22px; border-left: 4px solid #F59E0B;
        margin-top: 14px; margin-bottom: 26px; }
    .analysis-block h4 { margin-top: 0; color: #F59E0B !important; font-size: 1.05rem; margin-bottom: 12px; }
    .analysis-block p { margin-bottom: 9px; color: #D1D5DB; font-size: 0.93rem; text-align: justify; }
    .analysis-block b.tag { color: #FBBF24; }
    .card { background-color: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 16px 18px; margin-bottom: 12px; }
    .card .t { color: #F9FAFB; font-weight: 700; margin-bottom: 4px; } .card .d { color: #D1D5DB; font-size: 0.9rem; line-height: 1.5; }
    .stTabs [data-baseweb="tab"] { padding: 10px 16px; background-color: #111827; border-radius: 5px; color: #9CA3AF; }
    .stTabs [aria-selected="true"] { background-color: #3B82F6 !important; color: white !important; font-weight: bold; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- carga de datos
@st.cache_data
def leer_csv(nombre, **kw):
    ruta = os.path.join(DATA, nombre)
    return pd.read_csv(ruta, **kw) if os.path.exists(ruta) else None


@st.cache_data
def leer_json(nombre):
    ruta = os.path.join(DATA, nombre)
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def leer_series():
    df = leer_csv("series_30hz.csv.gz")
    return df


R = leer_json("resumen_proyecto.json")


def fmt_i(x):
    return f"{int(round(x)):,}".replace(",", ".")


def fmt(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


# --------------------------------------------------------------------------- utilidades de interfaz
def kpi(label, value, sub, clase=""):
    st.markdown(f'<div class="metric-box {clase}"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)


def tarjeta(titulo, texto, color="#3B82F6"):
    st.markdown(f'<div class="card" style="border-left:4px solid {color};"><div class="t">{titulo}</div>'
                f'<div class="d">{texto}</div></div>', unsafe_allow_html=True)


def mostrar_img(ruta):
    if not os.path.exists(ruta):
        st.warning(f"No se encontró la imagen: {os.path.basename(ruta)}")
        return
    try:
        st.image(ruta, use_container_width=True)
    except TypeError:  # versiones antiguas de Streamlit
        st.image(ruta, use_column_width=True)


def bloque(titulo, muestra, lectura, conclusion, color="#F59E0B"):
    st.markdown(f"""
    <div class="analysis-block" style="border-left-color:{color};">
        <h4 style="color:{color} !important;">{titulo}</h4>
        <p><b class="tag">Qué muestra.</b> {muestra}</p>
        <p><b class="tag">Cómo leerlo.</b> {lectura}</p>
        <p><b class="tag">Conclusión.</b> {conclusion}</p>
    </div>""", unsafe_allow_html=True)


def figura(clave):
    info = FIGURAS[clave]
    mostrar_img(os.path.join(FIGDIR, info["archivo"]))
    bloque("Interpretación — " + info["titulo"], info["muestra"], info["lectura"], info["conclusion"])


def figuras_en_pestanas(seccion):
    claves = SECCIONES_FIGURAS[seccion]
    tabs = st.tabs([FIGURAS[c]["titulo"] for c in claves])
    for t, c in zip(tabs, claves):
        with t:
            figura(c)


def estilo_plotly(fig, alto=420):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.6)",
                      height=alto, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    return fig


def mostrar_plot(fig, alto=420):
    st.plotly_chart(estilo_plotly(fig, alto), use_container_width=True)


def aviso_faltan_datos(nombre):
    st.warning(f"No se encontró `demo/data/{nombre}`. Genera los artefactos con `src/pipeline_mejorado/exportar_artefactos_dashboard.py`.")


# --------------------------------------------------------------------------- barra lateral
with st.sidebar:
    st.markdown("### 🎼 DIRECCIÓN ORQUESTAL")
    st.markdown("<p style='color: #6B7280; font-size: 0.8rem;'>Análisis No Supervisado (DAT-255)</p>", unsafe_allow_html=True)
    st.markdown("---")
    seccion_actual = option_menu(
        menu_title="SECCIONES",
        options=["Panorama General", "Calidad de Datos", "Exploración de Variables", "Modelos & Selección de k",
                 "Clusters", "Directores & Obras", "Dinámica Temporal", "Informe & Galería", "Bitácora MLOps",
                 "Metodología & Gobernanza"],
        icons=["house", "shield-check", "search", "cpu", "bullseye", "people", "clock-history", "file-earmark-text",
               "journal-text", "gear"],
        menu_icon="cast", default_index=0,
        styles={"container": {"padding": "0!important", "background-color": "#111827"},
                "icon": {"color": "#9CA3AF", "font-size": "16px"},
                "nav-link": {"font-size": "14px", "text-align": "left", "margin": "4px 0", "color": "#9CA3AF"},
                "nav-link-selected": {"background-color": "#1E3A8A", "color": "white", "font-weight": "bold"}})
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #1F2937; padding: 15px; border-radius: 8px; border-left: 3px solid #10B981;">
        <div style="font-size: 0.7rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1px;">Modelo Activo</div>
        <div style="color: #10B981; font-weight: bold; font-size: 1.05rem; margin-top: 5px;">🟢 K-Means (k=5) · marco corporal</div>
        <div style="font-size: 0.75rem; color: #6B7280; margin-top: 5px;">v2.0.0 · 54 tomas · 26 variables</div>
    </div>""", unsafe_allow_html=True)

if st.session_state.get("ultima_seccion") != seccion_actual:
    st.session_state["ultima_seccion"] = seccion_actual
    registrar_evento("vista_seccion", {"seccion": seccion_actual})

st.markdown(f"""
<div class="top-header">
    <div>
        <h1 class="header-title">{seccion_actual}</h1>
        <p class="header-subtitle">Exploración analítica y descubrimiento de patrones cinemáticos en la dirección orquestal</p>
    </div>
    <div><span class="status-badge">● Sistema Operativo</span></div>
</div>""", unsafe_allow_html=True)


# =========================================================================== SECCIONES
def seccion_panorama():
    st.markdown("""
    ### Resumen Ejecutivo del Estudio
    La dirección orquestal es un acto de comunicación complejo. Con captura de movimiento (mocap) de alta precisión, este
    proyecto convierte el gesto del director en datos cuantificables y aplica **aprendizaje no supervisado** para descubrir
    patrones cinemáticos recurrentes, **sin etiquetas previas**. Se analizan **62 archivos C3D** de seis directores que
    interpretan fragmentos de Mozart, Dvořák y Bartók.
    """)
    st.info("Esta versión 2.0 reemplaza la demostración anterior, que se calculó sobre **una sola grabación** (k=3, Ward). "
            "Todos los números de este panel provienen de las **54 tomas únicas** completas.")
    if not R:
        aviso_faltan_datos("resumen_proyecto.json")
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Tomas únicas", R["tomas_unicas"], f"de {R['archivos']} archivos (8 pares fusionados)")
    with c2:
        kpi("Fotogramas válidos", fmt_i(R["frames_validos"]), f"{fmt(R['pct_validos'] * 100, 1)} % de {fmt_i(R['frames_tomas'])} · 120 Hz", "green")
    with c3:
        kpi("Clusters (k)", R["k_final"], "K-Means · elegido por estabilidad", "purple")
    with c4:
        kpi("Estabilidad (ARI)", fmt(R["ari_k5"]), "mitades de tomas · k=5", "orange")
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        kpi("Silhouette (k=5)", fmt(R["silhouette_k5"], 3), "estructura débil: continuo", "red")
    with c6:
        kpi("Davies-Bouldin (k=5)", fmt(R["db_k5"]), "mínimo entre los k evaluados")
    with c7:
        kpi("Duración analizada", f"{fmt(R['horas'])} h", f"{R['variables']} variables biomecánicas", "green")
    with c8:
        kpi("NMI con director", fmt(R["nmi_director_k5"]), f"con la obra: {fmt(R['nmi_obra_k5'])}", "purple")

    st.markdown("### Hallazgos clave")
    cols = st.columns(len(HALLAZGOS))
    for col, (t, d), color in zip(cols, HALLAZGOS, ["#3B82F6", "#EF4444", "#8B5CF6", "#F59E0B"]):
        with col:
            tarjeta(t, d, color)

    st.markdown("### Los cinco regímenes de movimiento")
    for nombre in NOMBRES:
        tarjeta(nombre, DESCRIPCION_CLUSTERS[nombre], COLORES[nombre])
    st.caption("Son regímenes **descriptivos** del espacio cinemático; no se han validado contra etiquetas musicales.")

    st.markdown("---")
    st.markdown("### Muestra de fotogramas procesados")
    st.write("Cada fila es un fotograma (1/120 s). Las columnas son variables biomecánicas en el marco corporal del director: "
             "velocidad, aceleración, posición lateral/frontal/vertical respecto del torso, ángulos de codo, etc.")
    muestra = leer_csv("muestra_fotogramas.csv.gz")
    if muestra is not None:
        cols_vista = ["recording_id", "obra", "director", "tiempo_seg", "cluster_nombre", "mano_izq_velocidad",
                      "mano_der_velocidad", "angulo_codo_izq", "angulo_codo_der", "velocidad_media_manos"]
        st.dataframe(muestra[cols_vista].head(15), use_container_width=True)


def seccion_calidad():
    st.markdown("""
    ### Auditoría de calidad de los datos
    Antes de modelar se auditó la captura. Este análisis cambió el proyecto: reveló **archivos duplicados**, **huecos de
    oclusión largos** y un **error de relleno** en el pipeline anterior. Cada pestaña muestra una evidencia con su interpretación.
    """)
    figuras_en_pestanas("calidad")
    st.markdown("---")
    st.markdown("### Exploración interactiva")
    c1, c2 = st.columns(2)
    with c1:
        occ = leer_csv("oclusion_por_marcador.csv")
        if occ is not None:
            occ = occ.sort_values("fraccion_ocluida", ascending=False)
            fig = px.bar(occ, x="marcador", y="fraccion_ocluida", title="Fracción media de fotogramas ocluidos por marcador (62 archivos)")
            fig.update_traces(marker_color="#3B82F6")
            fig.update_yaxes(tickformat=".0%")
            mostrar_plot(fig, 380)
        else:
            aviso_faltan_datos("oclusion_por_marcador.csv")
    with c2:
        fu = leer_csv("fusion_variantes.csv")
        if fu is not None:
            largo = fu.melt(id_vars="toma", value_vars=["variante_1", "variante_2", "fusionada"], var_name="version", value_name="pct_validos")
            fig = px.bar(largo, x="toma", y="pct_validos", color="version", barmode="group",
                         title="Fotogramas válidos antes y después de fusionar variantes (%)")
            fig.update_xaxes(tickangle=-40)
            mostrar_plot(fig, 380)
    tomas = leer_csv("tomas_resumen.csv")
    if tomas is not None:
        fig = px.bar(tomas.sort_values("pct_validos"), x="toma", y="pct_validos", color="director",
                     title="Porcentaje de fotogramas válidos por toma", hover_data=["obra", "frames"])
        fig.update_xaxes(tickangle=-90, tickfont=dict(size=8))
        mostrar_plot(fig, 420)
    rep = leer_json("reporte_limpieza_54_tomas.json")
    if rep:
        st.markdown("#### Reporte de limpieza por toma y marcador")
        toma = st.selectbox("Toma", sorted(rep.keys()), key="rep_toma")
        df = pd.DataFrame(rep[toma]).T
        df.index.name = "marcador"
        st.dataframe(df, use_container_width=True)
        st.caption("`ocluidos`: fotogramas con ceros XYZ · `huecos`: número de huecos · `frames_interpolados`: relleno de huecos ≤ 12 frames · "
                   "`residual_nan`: fotogramas que siguen sin dato (huecos largos).")


def seccion_exploracion():
    st.markdown("""
    ### Exploración de variables
    Cómo se distribuyen las 26 variables biomecánicas, cuánto se parecen entre sí y cuántas dimensiones útiles hay.
    """)
    figuras_en_pestanas("exploracion")
    st.markdown("---")
    st.markdown("### Distribución interactiva de una variable")
    muestra = leer_csv("muestra_fotogramas.csv.gz")
    if muestra is None:
        aviso_faltan_datos("muestra_fotogramas.csv.gz")
        return
    excluir = {"recording_id", "obra", "director", "tiempo_seg", "cluster", "cluster_suav", "cluster_nombre", "pc1", "pc2", "pc3", "tsne1", "tsne2"}
    variables = [c for c in muestra.columns if c not in excluir]
    c1, c2, c3 = st.columns([2, 1, 1])
    var = c1.selectbox("Variable", variables, index=variables.index("velocidad_media_manos") if "velocidad_media_manos" in variables else 0)
    agrupar = c2.selectbox("Separar por", ["Ninguno", "Cluster", "Director", "Obra"])
    pct = c3.slider("Recortar percentil superior", 90, 100, 99)
    d = muestra[[var, "cluster_nombre", "director", "obra"]].dropna()
    tope = np.percentile(d[var], pct)
    d = d[d[var] <= tope]
    color = {"Ninguno": None, "Cluster": "cluster_nombre", "Director": "director", "Obra": "obra"}[agrupar]
    fig = px.histogram(d, x=var, color=color, nbins=60, barmode="overlay", opacity=0.65, histnorm="probability density",
                       color_discrete_map=COLORES if color == "cluster_nombre" else None,
                       title=f"Distribución de {var} (muestra de {fmt_i(len(d))} fotogramas)")
    mostrar_plot(fig, 440)
    st.caption("Muestra aleatoria de 20.000 fotogramas válidos. Velocidades en m/s, aceleraciones en m/s², posiciones en m, ángulos en grados.")
    corr = muestra[variables].corr(method="spearman")
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Correlación de Spearman (muestra)")
    mostrar_plot(fig, 620)


def seccion_modelos():
    st.markdown("""
    ### Selección del algoritmo y del número de clusters
    Se compararon **K-Means**, **Ward aproximado** y **mezclas gaussianas** con k de 2 a 8, sobre tres conjuntos de variables.
    Ningún índice interno decide por sí solo, así que se combinó con un criterio de **estabilidad por mitades de tomas**.
    """)
    figuras_en_pestanas("modelos")
    st.markdown("---")
    st.markdown("### Explorador de métricas")
    met = leer_csv("metricas_k.csv")
    if met is None:
        aviso_faltan_datos("metricas_k.csv")
        return
    c1, c2, c3 = st.columns(3)
    conjuntos = c1.multiselect("Conjunto de variables", ["BASE", "MEJ", "VENT"], default=["BASE", "MEJ"])
    algoritmos = c2.multiselect("Algoritmo", ["KMeans", "Ward_aprox"], default=["KMeans"])
    metrica = c3.selectbox("Métrica", ["silhouette", "davies_bouldin", "calinski_harabasz", "inercia"])
    d = met[met.conjunto.isin(conjuntos) & met.algoritmo.isin(algoritmos)].copy()
    if len(d):
        d["serie"] = d.conjunto + " · " + d.algoritmo
        fig = px.line(d, x="k", y=metrica, color="serie", markers=True, title=f"{metrica} según k")
        mostrar_plot(fig, 400)
    else:
        st.info("Selecciona al menos un conjunto y un algoritmo.")
    st.caption("BASE = posiciones absolutas · MEJ = marco corporal (modelo final) · VENT = MEJ + ventana de 0,5 s. "
               "Mayor es mejor para silhouette y Calinski-Harabasz; menor es mejor para Davies-Bouldin.")
    c1, c2 = st.columns(2)
    with c1:
        est = leer_csv("estabilidad.csv")
        if est is not None:
            fig = px.box(est, x="k", y="ari", points="all", title="Estabilidad (ARI entre mitades de tomas)")
            fig.update_traces(marker_color="#10B981")
            mostrar_plot(fig, 380)
    with c2:
        gmm = leer_csv("gmm.csv")
        if gmm is not None:
            g = gmm.melt(id_vars="k", value_vars=["bic", "aic"], var_name="criterio", value_name="valor")
            fig = px.line(g, x="k", y="valor", color="criterio", markers=True, title="Mezclas gaussianas: BIC y AIC (sin mínimo claro)")
            mostrar_plot(fig, 380)
    tabla = d.drop(columns=["serie"], errors="ignore") if len(d) else met
    st.dataframe(tabla.round(4), use_container_width=True)
    st.markdown("""
    <div class="analysis-block" style="border-left-color:#10B981;">
        <h4 style="color:#10B981 !important;">¿Por qué k=5 y no k=2 o k=3?</h4>
        <p><b>k=2</b> tiene el mayor Silhouette (0,20) pero es muy inestable (ARI 0,28): el límite entre «lento» y «rápido» cambia según las tomas usadas.
        <b>k=3</b> era el valor de la demostración anterior, calculada sobre una sola grabación; con todos los datos tiene el menor Silhouette de los primeros k.
        <b>k=5</b> es el más estable (ARI 0,71) y tiene el menor Davies-Bouldin (1,79), con una resolución interpretable. La elección se justifica por estabilidad, no por Silhouette.</p>
    </div>""", unsafe_allow_html=True)


def seccion_clusters():
    st.markdown("""
    ### Los cinco regímenes de movimiento
    Resultado de K-Means (k=5) sobre las 26 variables en el marco corporal. Explora la proyección, compara perfiles y lee la interpretación de cada gráfico.
    """)
    muestra = leer_csv("muestra_fotogramas.csv.gz")
    if muestra is None:
        aviso_faltan_datos("muestra_fotogramas.csv.gz")
    else:
        st.markdown("### Proyección interactiva")
        c1, c2, c3 = st.columns(3)
        proy = c1.selectbox("Proyección", ["PCA (PC1 vs PC2)", "PCA (PC1 vs PC3)", "t-SNE"])
        color = c2.selectbox("Colorear por", ["Cluster", "Director", "Obra"])
        n = c3.slider("Puntos a mostrar", 1000, 6000, 3000, step=500)
        ejes = {"PCA (PC1 vs PC2)": ("pc1", "pc2"), "PCA (PC1 vs PC3)": ("pc1", "pc3"), "t-SNE": ("tsne1", "tsne2")}[proy]
        d = muestra.dropna(subset=list(ejes)).head(n)
        col = {"Cluster": "cluster_nombre", "Director": "director", "Obra": "obra"}[color]
        fig = px.scatter(d, x=ejes[0], y=ejes[1], color=col, opacity=0.6, hover_data=["recording_id", "velocidad_media_manos"],
                         color_discrete_map=COLORES if col == "cluster_nombre" else None,
                         category_orders={"cluster_nombre": NOMBRES})
        fig.update_traces(marker=dict(size=4))
        mostrar_plot(fig, 520)
        st.caption("La proyección a dos ejes retiene menos de la mitad de la varianza (PCA: 39,4 %); los clusters se solapan en los bordes.")

    st.markdown("### Perfil comparado")
    z = leer_csv("perfil_z.csv")
    if z is not None:
        z = z.set_index("variable")
        elegidos = st.multiselect("Clusters a comparar", NOMBRES, default=NOMBRES)
        grupos = {"Velocidad y aceleración": [v for v in z.index if "velocidad" in v or "aceleracion" in v],
                  "Posición (marco corporal)": [v for v in z.index if v.endswith(("_lat", "_fre", "_ver"))],
                  "Postura": [v for v in z.index if "angulo" in v or "dist_torso" in v or "asimetria" in v]}
        grupo = st.radio("Grupo de variables", list(grupos.keys()), horizontal=True)
        vars_ = grupos[grupo]
        fig = go.Figure()
        for n_ in elegidos:
            fig.add_trace(go.Scatterpolar(r=z.loc[vars_, n_].tolist() + [z.loc[vars_[0], n_]], theta=vars_ + [vars_[0]], name=n_,
                                          line=dict(color=COLORES[n_]), fill="toself", opacity=0.35))
        fig.update_layout(title="Perfil estandarizado (z-score)", polar=dict(radialaxis=dict(range=[-2, 2])))
        mostrar_plot(fig, 520)
    perfil = leer_csv("perfil_clusters.csv")
    if perfil is not None:
        st.markdown("#### Perfil en unidades físicas (medianas)")
        st.dataframe(perfil.set_index("cluster").T.round(2), use_container_width=True)
    st.markdown("---")
    figuras_en_pestanas("clusters")


def seccion_directores():
    st.markdown("""
    ### Directores y obras
    ¿Los clusters dependen de **quién** dirige o de **qué** se dirige? El resultado principal: el director explica
    bastante más variación que la obra, aunque la obra también deja huella.
    """)
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("NMI cluster–director", fmt(R.get("nmi_director_k5", 0)), "con posiciones absolutas era ≈ 0,52", "purple")
    with c2:
        kpi("NMI cluster–obra", fmt(R.get("nmi_obra_k5", 0)), "casi independiente de la obra", "orange")
    with c3:
        eta = R.get("eta2", {})
        kpi("Varianza explicada", f"{fmt(eta.get('director', 0) * 100, 0)} % / {fmt(eta.get('compositor', 0) * 100, 0)} %", "director / obra (velocidad por toma)", "green")
    od = leer_csv("ocupacion_director.csv")
    oo = leer_csv("ocupacion_obra.csv")
    c1, c2 = st.columns(2)
    for col, datos, nombre, eje in [(c1, od, "director", "Director"), (c2, oo, "obra", "Obra")]:
        with col:
            if datos is not None:
                largo = datos.melt(id_vars=nombre, var_name="cluster", value_name="porcentaje")
                fig = px.bar(largo, x=nombre, y="porcentaje", color="cluster", color_discrete_map=COLORES,
                             title=f"Composición por {eje.lower()} (% de fotogramas)", category_orders={"cluster": NOMBRES})
                mostrar_plot(fig, 420)
    tomas = leer_csv("tomas_resumen.csv")
    if tomas is not None:
        st.markdown("### Explorador de tomas")
        f1, f2 = st.columns(2)
        dirs = f1.multiselect("Director", sorted(tomas.director.unique()), default=sorted(tomas.director.unique()))
        obras = f2.multiselect("Obra", sorted(tomas.obra.unique()), default=sorted(tomas.obra.unique()))
        t = tomas[tomas.director.isin(dirs) & tomas.obra.isin(obras)]
        if len(t):
            fig = px.box(t, x="director", y="vel_media_manos_mediana", color="obra", points="all",
                         title="Mediana de la velocidad media de manos por toma (m/s)")
            mostrar_plot(fig, 420)
            st.dataframe(t.round(2), use_container_width=True)
        else:
            st.info("No hay tomas con ese filtro.")
    st.markdown("---")
    figuras_en_pestanas("directores")


def seccion_dinamica():
    st.markdown("""
    ### Dinámica temporal del gesto
    El modelo clasifica cada fotograma sin memoria, así que las etiquetas pueden alternar muy rápido. Un **suavizado de mayoría de 0,5 s**
    permite leer episodios. Elige una toma para ver su velocidad y su secuencia de regímenes.
    """)
    series = leer_series()
    if series is None:
        aviso_faltan_datos("series_30hz.csv.gz")
    else:
        tomas = sorted(series.recording_id.unique())
        c1, c2 = st.columns([3, 1])
        toma = c1.selectbox("Toma", tomas, index=tomas.index("Mozart_Conductor04_trial01") if "Mozart_Conductor04_trial01" in tomas else 0)
        modo = c2.radio("Etiquetas", ["Suavizadas", "Crudas"], horizontal=True)
        d = series[series.recording_id == toma].copy()
        tmax = float(d.tiempo_seg.max())
        ventana = st.slider("Ventana de tiempo (s)", 0.0, max(tmax, 1.0), (0.0, min(60.0, tmax)))
        d = d[(d.tiempo_seg >= ventana[0]) & (d.tiempo_seg <= ventana[1])]
        col_c = "cluster_suav" if modo == "Suavizadas" else "cluster"
        d["régimen"] = d[col_c].map(lambda i: NOMBRES[int(i) - 1])
        fig = px.line(d, x="tiempo_seg", y="velocidad_media_manos", title=f"Velocidad media de las manos — {toma}")
        fig.update_traces(line=dict(color="#9CA3AF", width=1))
        mostrar_plot(fig, 300)
        fig2 = px.scatter(d, x="tiempo_seg", y="régimen", color="régimen", color_discrete_map=COLORES,
                          category_orders={"régimen": NOMBRES}, title="Régimen activo (30 Hz)")
        fig2.update_traces(marker=dict(size=5, symbol="square"))
        mostrar_plot(fig2, 320)
        st.caption("Los tramos sin puntos son fotogramas sin dato válido (oclusiones largas).")

    dw = R.get("dwell", {})
    if dw:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("Cambios por segundo", fmt(dw["cambios_por_s_raw"], 1), "etiquetas crudas")
        with c2:
            kpi("Tras suavizar", fmt(dw["cambios_por_s_sm"], 1), "filtro de mayoría 0,5 s", "green")
        with c3:
            kpi("Episodio mediano", f"{fmt(dw['raw_median'], 2)} s", "sin suavizar", "orange")
        with c4:
            kpi("Episodio mediano", f"{fmt(dw['sm_median'], 2)} s", "suavizado", "purple")
    ep = leer_csv("episodios_suavizados.csv")
    tr = leer_csv("transiciones.csv")
    c1, c2 = st.columns(2)
    with c1:
        if ep is not None:
            st.markdown("#### Duración de episodios por régimen")
            st.dataframe(ep.round(3).rename(columns={"count": "episodios", "median": "mediana_s", "mean": "media_s"}), use_container_width=True)
    with c2:
        if tr is not None:
            m = tr.set_index(tr.columns[0])
            fig = px.imshow(m, color_continuous_scale="Blues", zmin=0, aspect="auto", title="Probabilidad de transición entre episodios")
            mostrar_plot(fig, 360)
    st.markdown("---")
    figuras_en_pestanas("dinamica")


def seccion_informe():
    st.markdown("""
    ### Informe técnico y galería de figuras
    El informe completo (más de 60 páginas) documenta metodología, resultados y limitaciones. Aquí puedes descargarlo y revisar
    las 32 figuras con su interpretación.
    """)
    c1, c2, c3 = st.columns(3)
    with c1:
        if os.path.exists(INFORME):
            with open(INFORME, "rb") as f:
                st.download_button("📄 Descargar informe (Word)", f.read(), file_name=os.path.basename(INFORME),
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                   on_click=registrar_evento, args=("descarga", {"archivo": "informe.docx"}))
        else:
            st.warning("El informe no está en `informe/`.")
    with c2:
        muestra = leer_csv("muestra_fotogramas.csv.gz")
        if muestra is not None:
            st.download_button("⬇️ Muestra de fotogramas (CSV)", muestra.to_csv(index=False).encode("utf-8"),
                               file_name="muestra_fotogramas.csv", mime="text/csv",
                               on_click=registrar_evento, args=("descarga", {"archivo": "muestra_fotogramas.csv"}))
    with c3:
        met = leer_csv("metricas_k.csv")
        if met is not None:
            st.download_button("⬇️ Métricas por k (CSV)", met.to_csv(index=False).encode("utf-8"), file_name="metricas_k.csv",
                               mime="text/csv", on_click=registrar_evento, args=("descarga", {"archivo": "metricas_k.csv"}))
    st.markdown("---")
    st.markdown("### Galería")
    etiquetas = {"metodologia": "Metodología", "calidad": "Calidad de datos", "exploracion": "Exploración", "modelos": "Modelos",
                 "clusters": "Clusters", "directores": "Directores y obras", "dinamica": "Dinámica"}
    sel = st.multiselect("Secciones", list(etiquetas.values()), default=list(etiquetas.values()))
    inv = {v: k for k, v in etiquetas.items()}
    for nombre in sel:
        st.markdown(f"## {nombre}")
        for clave in SECCIONES_FIGURAS[inv[nombre]]:
            with st.expander(f"{FIGURAS[clave]['titulo']}", expanded=False):
                figura(clave)


def seccion_bitacora():
    st.markdown("""
    ### Bitácora de análisis y trazabilidad (MLOps)
    Cada decisión del pipeline queda registrada para que sea reproducible y auditable. Se muestran los **hitos de ingeniería de datos**
    de la versión 2.0 y el **registro de eventos** de esta aplicación.
    """)
    if not R:
        aviso_faltan_datos("resumen_proyecto.json")
        return
    g = R.get("gaps", {})
    hitos = [
        ("HITO-01", "Ingesta y consolidación", f"{R['archivos']} archivos C3D de la Universidad de Edimburgo convertidos de mm a m y consolidados en {fmt_i(R['frames_originales'])} fotogramas.", "COMPLETADO", "#3B82F6"),
        ("HITO-02", "Auditoría y deduplicación", "Se detectaron 8 pares _1/_2 con coordenadas idénticas (98,6 %–100 %) y datos complementarios. Se fusionaron: quedan 54 tomas únicas.", "COMPLETADO", "#EF4444"),
        ("HITO-03", "Limpieza y tratamiento de señal", "Ceros XYZ → NaN. Interpolación solo de huecos ≤ 12 frames; los largos se conservan como NaN (corrige el relleno por vecino). Savitzky-Golay de 9 frames por tramos válidos.", "COMPLETADO", "#10B981"),
        ("HITO-04", "Ingeniería de variables", f"{R['variables']} variables en un marco corporal (lateral, frontal, vertical respecto del torso); se descartó la energía por redundancia con la velocidad.", "COMPLETADO", "#8B5CF6"),
        ("HITO-05", "Modelado y selección", f"K-Means, Ward aproximado y GMM con k=2..8. Elegido K-Means k={R['k_final']} por estabilidad (ARI {fmt(R['ari_k5'])}) y Davies-Bouldin ({fmt(R['db_k5'])}). Silhouette {fmt(R['silhouette_k5'], 3)} (débil).", "COMPLETADO", "#F59E0B"),
        ("HITO-06", "Evaluación, informe y despliegue", "Estabilidad por mitades de tomas, asociación con director/obra, dinámica temporal. Informe de 60+ páginas y este dashboard.", "EN PRODUCCIÓN", "#EC4899"),
    ]
    for hid, titulo, desc, estado, color in hitos:
        st.markdown(f"""
        <div style="background-color: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 18px; margin-bottom: 14px; border-left: 5px solid {color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div><span style="font-family: monospace; color: #6B7280; font-weight: bold;">[{hid}]</span>
                <strong style="color: #F9FAFB; font-size: 1.05rem; margin-left: 10px;">{titulo}</strong></div>
                <span style="color: {color}; font-size: 0.75rem; background: {color}20; padding: 4px 10px; border-radius: 12px; font-weight: bold; border: 1px solid {color}50;">{estado}</span>
            </div>
            <div style="color: #D1D5DB; font-size: 0.93rem; line-height: 1.5;">{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("### Registro de eventos de la aplicación")
    eventos = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for linea in f:
                try:
                    eventos.append(json.loads(linea))
                except ValueError:
                    continue
    if eventos:
        df = pd.DataFrame(eventos)
        df["detalles"] = df["detalles"].astype(str)
        st.dataframe(df.iloc[::-1].head(200), use_container_width=True)
    else:
        st.info("Todavía no hay eventos registrados.")
    with st.expander("🔍 Resumen técnico del proyecto (JSON)"):
        st.json(R)


def seccion_metodologia():
    st.markdown("""
    ### Marco de trabajo: CRISP-DM
    El proyecto sigue un ciclo completo de ciencia de datos. La naturaleza iterativa del proceso se notó: los hallazgos de la **evaluación**
    (los clusters dependían del director) obligaron a volver a la **preparación** para rediseñar las variables.
    """)
    st.markdown("""
    <div class="analysis-block" style="border-left-color:#3B82F6;">
        <h4 style="color:#3B82F6 !important;">Fases de desarrollo</h4>
        <ul>
            <li><b>1. Comprensión del negocio:</b> ¿hay regímenes de movimiento recurrentes en el gesto del director y cómo se distribuyen entre obras y directores?</li>
            <li><b>2. Comprensión de los datos:</b> 62 archivos C3D (120 Hz, 20 marcadores). Auditoría de oclusiones, huecos, marcadores ausentes y duplicados.</li>
            <li><b>3. Preparación:</b> fusión de variantes, interpolación de huecos ≤ 12 frames, Savitzky-Golay por tramos, marco corporal y 26 variables.</li>
            <li><b>4. Modelado:</b> K-Means, Ward aproximado y mezclas gaussianas para k=2..8; submuestreo a 30 Hz para el ajuste.</li>
            <li><b>5. Evaluación:</b> Silhouette, Davies-Bouldin, Calinski-Harabasz, BIC, estabilidad por mitades de tomas (ARI) y NMI con director, obra y toma.</li>
            <li><b>6. Despliegue:</b> scripts reproducibles, este dashboard, informe técnico y bitácora.</li>
        </ul>
    </div>""", unsafe_allow_html=True)
    st.markdown("### Qué cambió respecto de la versión 1.0")
    cambios = pd.DataFrame([
        ["Datos", "Primera grabación (15.063 fotogramas)", "54 tomas (752.537 fotogramas válidos)"],
        ["Archivos", "62 grabaciones independientes", "54 tomas; 8 pares fusionados"],
        ["Huecos largos", "Relleno por vecino (posición congelada)", "Se conservan como NaN"],
        ["Posiciones", "Absolutas en el laboratorio", "Marco corporal (lateral, frontal, vertical)"],
        ["Variables", "12 (con energía cinética y batuta)", "26 (sin energía; batuta aparte)"],
        ["Elección de k", "Silhouette → k=3 (Ward)", "Estabilidad + varios índices → k=5 (K-Means)"],
        ["Interpretación", "Legato / staccato (no validado)", "Regímenes cinemáticos descriptivos"],
    ], columns=["Aspecto", "Versión 1.0", "Versión 2.0"])
    st.dataframe(cambios, use_container_width=True, hide_index=True)
    st.markdown("### Limitaciones y gobernanza")
    for t, d in [
        ("Descriptivo, no musical", "Los clusters son regímenes cinemáticos. No hay etiquetas de compás, dinámica o tempo para validarlos; asociarlos a «legato» o «staccato» requeriría esa validación."),
        ("Estilo individual", "Parte de la estructura refleja diferencias entre directores (estilo y anatomía). Las comparaciones entre obras son más fiables dentro de cada director."),
        ("Estructura débil", "Silhouette ≤ 0,20 y BIC sin mínimo: el gesto se parece más a un continuo que a estados discretos. C5 (intenso) es el menos cohesivo."),
        ("Datos faltantes", "Solo se agrupan fotogramas completos; las tomas con más oclusión aportan menos datos."),
        ("Representatividad", "Seis directores y tres obras: no se generaliza más allá del conjunto."),
    ]:
        tarjeta(t, d, "#F59E0B")
    st.markdown("### Reproducibilidad")
    st.code("""# 1) Dependencias
pip install -r requirements.txt
# 2) Pipeline mejorado (ver src/pipeline_mejorado/README.md)
python src/pipeline_mejorado/load_raw.py
python src/pipeline_mejorado/mocap_pipeline.py
python src/pipeline_mejorado/modelado_a.py && python src/pipeline_mejorado/modelado_b.py && python src/pipeline_mejorado/analisis_c.py
# 3) Artefactos del dashboard y ejecución
python src/pipeline_mejorado/exportar_artefactos_dashboard.py
streamlit run app.py""", language="bash")
    st.markdown("### Figuras de apoyo")
    figuras_en_pestanas("metodologia")


# --------------------------------------------------------------------------- enrutador
RUTAS = {
    "Panorama General": seccion_panorama,
    "Calidad de Datos": seccion_calidad,
    "Exploración de Variables": seccion_exploracion,
    "Modelos & Selección de k": seccion_modelos,
    "Clusters": seccion_clusters,
    "Directores & Obras": seccion_directores,
    "Dinámica Temporal": seccion_dinamica,
    "Informe & Galería": seccion_informe,
    "Bitácora MLOps": seccion_bitacora,
    "Metodología & Gobernanza": seccion_metodologia,
}
RUTAS[seccion_actual]()
