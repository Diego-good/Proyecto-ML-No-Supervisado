#!/usr/bin/env python3
"""Pipeline de mocap: limpieza, features biomecánicas y clustering no supervisado.

Uso demo CSV: python features_clustering.py --input ../Persona_1/muestra_datos.csv --format csv
Uso dataset completo: python features_clustering.py --input ../Persona_1/datos_crudos.parquet --format parquet
CSV esperado: separador ';' y coma decimal (se detecta automáticamente).

Cambio respecto a la versión anterior: las métricas (silhouette, Davies-Bouldin) se evalúan
sobre una muestra ESTRATIFICADA por cluster, de modo que ningún cluster (por pequeño que sea)
desaparece de la muestra de evaluación.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

FPS_DEFAULT = 120.0
MARKERS = ['LFHD','RFHD','LBHD','RBHD','LSHO','RSHO','LELB','RELB','LWRA','LWRB','RWRA','RWRB','LFIN','RFIN','Baton End','Baton Shaft','C7','CLAV','STRN','T10']


def read_data(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == 'parquet' or path.suffix.lower() == '.parquet':
        try:
            return pd.read_parquet(path)
        except ImportError as e:
            raise SystemExit('Para leer Parquet instala pyarrow: pip install pyarrow. Alternativamente usa muestra_datos.csv.') from e
    # El CSV de entrega usa ; y coma decimal.
    try:
        return pd.read_csv(path, sep=';', decimal=',')
    except Exception:
        return pd.read_csv(path)


def marker_cols(df):
    return [f'{m}_{a}' for m in MARKERS for a in 'xyz' if f'{m}_{a}' in df.columns]


def clean_recording(g: pd.DataFrame, fps: float, max_gap: int = 12) -> tuple[pd.DataFrame, dict]:
    """Ceros XYZ simultáneos se interpretan como oclusión; interpola huecos cortos,
    rellena bordes por vecino y suaviza con Savitzky-Golay sin cruzar grabaciones."""
    g = g.sort_values('frame').copy().reset_index(drop=True)
    cols = marker_cols(g)
    missing_before = int(g[cols].isna().sum().sum())
    zero_mask = {}
    for m in MARKERS:
        xyz = [f'{m}_{a}' for a in 'xyz' if f'{m}_{a}' in g]
        if len(xyz) == 3:
            zero = (g[xyz].abs().sum(axis=1) == 0)
            zero_mask[m] = int(zero.sum())
            g.loc[zero, xyz] = np.nan
    # Los faltantes estructurales completos se mantienen como NaN si no hay datos.
    for c in cols:
        g[c] = g[c].interpolate(method='linear', limit=max_gap, limit_direction='both')
        g[c] = g[c].ffill().bfill()
    # Si un marcador no existe en toda la grabación, queda NaN; no inventar anatomía.
    for c in cols:
        x = g[c].to_numpy(dtype=float)
        valid = np.isfinite(x)
        if valid.sum() < 5:
            continue
        win = nine_window(len(x))
        if win >= 5:
            try: g[c] = savgol_filter(x, window_length=win, polyorder=2, mode='interp')
            except Exception: pass
    return g, {'n_filas':len(g), 'faltantes_iniciales':missing_before, 'ceros_a_nan_por_marcador':zero_mask,
               'faltantes_finales':int(g[cols].isna().sum().sum())}


def nine_window(n):
    # Suavizado de 0.075 s a 120 Hz, impar y nunca mayor que la serie.
    w = max(5, int(round(FPS_DEFAULT*0.075)) | 1)
    if w > n: w = n if n % 2 else n-1
    return w


def xyz(g, marker):
    cs=[f'{marker}_{a}' for a in 'xyz']
    if all(c in g.columns for c in cs):
        return g[cs].to_numpy(float)
    return None


def add_vec_features(out, name, pos, dt):
    if pos is None: return
    vel=np.gradient(pos, dt, axis=0)
    acc=np.gradient(vel, dt, axis=0)
    out[f'{name}_velocidad'] = np.linalg.norm(vel,axis=1)
    out[f'{name}_aceleracion'] = np.linalg.norm(acc,axis=1)
    out[f'{name}_energia_cinetica_rel'] = 0.5*np.sum(vel**2,axis=1) # masa unitaria
    out[f'{name}_pos_x']=pos[:,0]; out[f'{name}_pos_y']=pos[:,1]; out[f'{name}_pos_z']=pos[:,2]


def angle_3pt(a,b,c):
    u=a-b; v=c-b
    den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
    cos=np.divide(np.sum(u*v,axis=1),den,out=np.full(len(den),np.nan),where=den>1e-10)
    return np.degrees(np.arccos(np.clip(cos,-1,1)))


def extract_features(g: pd.DataFrame, fps: float) -> pd.DataFrame:
    dt=1.0/fps
    f=pd.DataFrame(index=g.index)
    for key,m in [('mano_izq','LWRA'),('mano_der','RWRA'),('codo_izq','LELB'),('codo_der','RELB'),('batuta','Baton End')]:
        p=xyz(g,m)
        if p is not None: add_vec_features(f,key,p,dt)
    # Centro de torso como referencia corporal para reducir variación de posición global.
    torso_candidates=[xyz(g,m) for m in ['C7','CLAV','STRN','T10']]
    torso=[p for p in torso_candidates if p is not None and np.isfinite(p).any()]
    if torso:
        torso_ref=np.nanmean(np.stack(torso),axis=0)
        for key,m in [('mano_izq','LWRA'),('mano_der','RWRA')]:
            p=xyz(g,m)
            if p is not None:
                rel=p-torso_ref
                f[f'{key}_dist_torso']=np.linalg.norm(rel,axis=1)
    # Ángulos articulares (grados).
    for side,elbow,wrist,shoulder in [('izq','LELB','LWRA','LSHO'),('der','RELB','RWRA','RSHO')]:
        a,b,c=xyz(g,shoulder),xyz(g,elbow),xyz(g,wrist)
        if a is not None and b is not None and c is not None:
            f[f'angulo_codo_{side}']=angle_3pt(a,b,c)
    # Orientación de batuta respecto a eje horizontal XY.
    be,bs=xyz(g,'Baton End'),xyz(g,'Baton Shaft')
    if be is not None and bs is not None:
        d=be-bs
        f['angulo_batuta_xy']=np.degrees(np.arctan2(d[:,1],d[:,0]))
        f['longitud_batuta']=np.linalg.norm(d,axis=1)
    # Intensidad global de movimiento de manos y asimetría.
    vl=f.get('mano_izq_velocidad'); vr=f.get('mano_der_velocidad')
    if vl is not None and vr is not None:
        f['velocidad_media_manos']=(vl+vr)/2
        f['asimetria_velocidad_manos']=np.abs(vl-vr)
    # Identificadores conservados para trazabilidad.
    for c in ['recording_id','frame','tiempo_seg']:
        if c in g: f[c]=g[c].values
    return f.replace([np.inf,-np.inf],np.nan)


def stratified_idx(labels, n_sample, min_per_cluster=30, seed=42):
    """Índices de una muestra estratificada por cluster.

    - Si n <= n_sample devuelve todos los índices.
    - Cada cluster recibe una cuota proporcional a su tamaño, pero con un mínimo de
      `min_per_cluster` (o todo el cluster si es más pequeño). Así los clusters pequeños
      nunca desaparecen de la muestra.
    - La muestra puede superar ligeramente n_sample por culpa de los mínimos.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if n <= n_sample:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(labels, return_counts=True)
    quota = np.round(counts / n * n_sample).astype(int)
    quota = np.maximum(quota, np.minimum(min_per_cluster, counts))
    quota = np.minimum(quota, counts)
    idx = []
    for u, q in zip(uniq, quota):
        members = np.flatnonzero(labels == u)
        idx.append(members if q >= len(members) else rng.choice(members, q, replace=False))
    return np.sort(np.concatenate(idx))


def safe_metrics(Ze, le):
    """(silhouette, davies_bouldin) o (nan, nan) si no hay entre 2 y n-1 clusters."""
    n_lab = len(np.unique(le))
    if n_lab < 2 or n_lab > len(le) - 1:
        return np.nan, np.nan
    return silhouette_score(Ze, le), davies_bouldin_score(Ze, le)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True); ap.add_argument('--format',choices=['auto','csv','parquet'],default='auto')
    ap.add_argument('--fps',type=float,default=FPS_DEFAULT); ap.add_argument('--k-min',type=int,default=2); ap.add_argument('--k-max',type=int,default=6)
    ap.add_argument('--metric-sample',type=int,default=3000, help='Tamaño objetivo de la muestra estratificada para métricas')
    ap.add_argument('--min-per-cluster',type=int,default=30, help='Mínimo de puntos por cluster en la muestra de evaluación')
    ap.add_argument('--ward-sample',type=int,default=3500, help='Tamaño de la muestra para entrenar Ward')
    ap.add_argument('--outdir',default='salida_features_clustering')
    args=ap.parse_args(); outdir=Path(args.outdir); figdir=outdir/'figuras'; figdir.mkdir(parents=True,exist_ok=True)
    df=read_data(Path(args.input),args.format)
    if 'recording_id' not in df: df['recording_id']='grabacion_1'
    cleaned=[]; features=[]; clean_report={}
    for rid,g in df.groupby('recording_id',sort=False):
        gc,rep=clean_recording(g,args.fps); clean_report[str(rid)]=rep
        ff=extract_features(gc,args.fps)
        features.append(ff)
        cleaned.append(gc)
    clean=pd.concat(cleaned,ignore_index=True); feat=pd.concat(features,ignore_index=True)
    meta=[c for c in ['recording_id','frame','tiempo_seg'] if c in feat]
    numeric=[c for c in feat.select_dtypes(include=np.number).columns if c not in ['frame','tiempo_seg']]
    # Eliminar features completamente vacías o constantes.
    X=feat[numeric].copy().replace([np.inf,-np.inf],np.nan)
    X=X.dropna(axis=1,how='all')
    X=X.loc[:,X.nunique(dropna=True)>1]
    X=X.interpolate(limit_direction='both').ffill().bfill()
    valid_cols=X.columns.tolist()
    if len(valid_cols)<2: raise SystemExit('No hay suficientes features numéricas válidas para clustering.')
    scaler=StandardScaler(); Z=scaler.fit_transform(X)
    n=len(Z)
    rows=[]; fitted={}
    for k in range(args.k_min,args.k_max+1):
        if k>=n: continue
        # --- KMeans: ajuste sobre todos los datos, evaluación sobre muestra estratificada ---
        km=KMeans(n_clusters=k,random_state=42,n_init=10); lab=km.fit_predict(Z)
        eidx=stratified_idx(lab,args.metric_sample,args.min_per_cluster)
        sil,db=safe_metrics(Z[eidx],lab[eidx])
        rows.append({'algoritmo':'KMeans','k':k,'silhouette':sil,'davies_bouldin':db,'inercia':km.inertia_,
                     'n_eval':len(eidx),'tamanos_cluster':np.bincount(lab,minlength=k).tolist()})
        fitted[('KMeans',k)]=(lab,km)
        # --- Ward aproximado: O(n^2), se entrena en un subconjunto ---
        # El subconjunto también se estratifica (usando las etiquetas de KMeans) para que
        # los grupos pequeños estén representados al construir la jerarquía.
        hidx=stratified_idx(lab,args.ward_sample,args.min_per_cluster)
        ag=AgglomerativeClustering(n_clusters=k,linkage='ward'); htrain=ag.fit_predict(Z[hidx])
        centers=np.vstack([Z[hidx][htrain==j].mean(axis=0) for j in range(k)])
        la=np.argmin(((Z[:,None,:]-centers[None,:,:])**2).sum(axis=2),axis=1)
        eidx_a=stratified_idx(la,args.metric_sample,args.min_per_cluster)
        sil,db=safe_metrics(Z[eidx_a],la[eidx_a])
        rows.append({'algoritmo':'Jerarquico_Ward_aprox','k':k,'silhouette':sil,'davies_bouldin':db,'inercia':np.nan,
                     'n_eval':len(eidx_a),'tamanos_cluster':np.bincount(la,minlength=k).tolist()})
        fitted[('Jerarquico_Ward_aprox',k)]=(la,ag)
    metrics=pd.DataFrame(rows)
    # Selección automática: mayor silhouette, desempate por menor DB (ignorando modelos degenerados).
    valid=metrics.dropna(subset=['silhouette'])
    if valid.empty:
        raise SystemExit('Ningún modelo produjo ≥2 clusters. Revisa la limpieza (outliers/ceros) o usa más datos.')
    best=valid.sort_values(['silhouette','davies_bouldin'],ascending=[False,True]).iloc[0]
    alg,k=best['algoritmo'],int(best['k']); labels=fitted[(alg,k)][0]
    result=feat.copy(); result['cluster']=labels
    # Guardar resultados.
    result.to_csv(outdir/'features_clustering.csv',index=False)
    metrics.assign(tamanos_cluster=metrics['tamanos_cluster'].astype(str)).to_csv(outdir/'metricas_clustering.csv',index=False)
    clean_report['configuracion']={'fps':args.fps,'features_usadas':valid_cols,'algoritmo_seleccionado':alg,'k_seleccionado':k,
                                   'silhouette_seleccion':float(best['silhouette']),'davies_bouldin_seleccion':float(best['davies_bouldin']),
                                   'muestreo_metricas':'estratificado por cluster',
                                   'metricas_calculadas_sobre_muestra':int(best['n_eval']),
                                   'min_por_cluster':args.min_per_cluster,
                                   'tamanos_cluster_seleccionado':best['tamanos_cluster'],'filas_totales':n}
    (outdir/'reporte_limpieza_metricas.json').write_text(json.dumps(clean_report,indent=2,ensure_ascii=False),encoding='utf-8')
    # Figura 1: métricas por algoritmo/k.
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for name,sub in valid.groupby('algoritmo'):
        axs[0].plot(sub.k,sub.silhouette,marker='o',label=name); axs[1].plot(sub.k,sub.davies_bouldin,marker='o',label=name)
    axs[0].set(title='Silhouette (mayor es mejor)',xlabel='Número de clusters k',ylabel='Silhouette')
    axs[1].set(title='Davies-Bouldin (menor es mejor)',xlabel='Número de clusters k',ylabel='Davies-Bouldin')
    for ax in axs: ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(figdir/'01_metricas_modelos.png',dpi=160); plt.close(fig)
    # Figura 2: PCA sobre muestra estratificada del modelo seleccionado.
    pidx=stratified_idx(labels,args.metric_sample,args.min_per_cluster)
    pca=PCA(n_components=2,random_state=42); coords=pca.fit_transform(Z[pidx])
    fig,ax=plt.subplots(figsize=(8,6)); sc=ax.scatter(coords[:,0],coords[:,1],c=labels[pidx],s=8,cmap='tab10',alpha=.65)
    ax.set(title=f'PCA de features | {alg}, k={k}',xlabel=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',ylabel=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    fig.colorbar(sc,ax=ax,label='Cluster'); fig.tight_layout(); fig.savefig(figdir/'02_pca_clusters.png',dpi=160); plt.close(fig)
    # Figura 3: evolución temporal de cluster, primer recording.
    first=result[result.recording_id==result.recording_id.iloc[0]] if 'recording_id' in result else result
    if 'tiempo_seg' in first:
        fig,ax=plt.subplots(figsize=(12,3.8)); ax.scatter(first.tiempo_seg,first.cluster,c=first.cluster,cmap='tab10',s=4)
        ax.set(title='Secuencia temporal de clusters',xlabel='Tiempo (s)',ylabel='Cluster'); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(figdir/'03_clusters_tiempo.png',dpi=160); plt.close(fig)
    # Figura 4: perfiles medios estandarizados por cluster.
    prof=pd.DataFrame(Z,columns=valid_cols).assign(cluster=labels).groupby('cluster').mean()
    top=prof.var(axis=0).sort_values(ascending=False).head(min(12,len(valid_cols))).index
    fig,ax=plt.subplots(figsize=(12,5)); prof[top].T.plot(kind='bar',ax=ax)
    ax.set(title='Perfil medio estandarizado por cluster (features más discriminantes)',ylabel='Media z-score',xlabel='Feature'); ax.tick_params(axis='x',rotation=45)
    fig.tight_layout(); fig.savefig(figdir/'04_perfiles_clusters.png',dpi=160); plt.close(fig)
    print(json.dumps({'filas':n,'n_features':len(valid_cols),'metricas':metrics.to_dict(orient='records'),'seleccion':{'algoritmo':alg,'k':k},'outdir':str(outdir)},ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':
    main()