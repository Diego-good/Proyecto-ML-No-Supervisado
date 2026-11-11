"""Exporta los artefactos compactos que consume el dashboard (app.py).
Entradas (carpeta work/): D_labels.pkl, features_all.pkl, metricas_k.pkl, estabilidad.pkl, gmm.pkl, nmi.pkl, sens.pkl,
tabla_clusters.pkl, perfil_z.pkl, ct_director.pkl, ct_comp.pkl, trans_p.pkl, runs_sm.pkl, tomas.pkl, fusion.json, clean_info.json,
res_c.json, sil_k5.json, pca_var.json, sub_idx.npy, tsne_sub.npy, energy_corr.json.
Salida: demo/data/*"""
import json, numpy as np, pandas as pd, os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as NMI
from mocap_pipeline import FEATS_MEJ, MARKERS
OUT='demo/data/'; os.makedirs(OUT,exist_ok=True)
NAMES=['C1 Compacto','C2 Frontal moderado','C3 Amplio','C4 Asimétrico','C5 Intenso']
D=pd.read_pickle('work/D_labels.pkl'); rc=json.load(open('work/res_c.json'))
D['cluster']=D.lab_MEJ_5+1; D['cluster_suav']=D.lab_MEJ_5_suav+1
# --- métricas
m=pd.read_pickle('work/metricas_k.pkl').drop(columns=['tamanos']); m.to_csv(OUT+'metricas_k.csv',index=False,float_format='%.5f')
pd.read_pickle('work/estabilidad.pkl').to_csv(OUT+'estabilidad.csv',index=False,float_format='%.5f')
pd.read_pickle('work/gmm.pkl').to_csv(OUT+'gmm.csv',index=False)
pd.read_pickle('work/nmi.pkl').to_csv(OUT+'nmi.csv',index=False,float_format='%.5f')
pd.read_pickle('work/sens.pkl').to_csv(OUT+'sensibilidad.csv',index=False,float_format='%.5f')
# --- perfiles
t=pd.read_pickle('work/tabla_clusters.pkl').reset_index(); t['cluster']=NAMES; t.to_csv(OUT+'perfil_clusters.csv',index=False,float_format='%.4f')
z=pd.read_pickle('work/perfil_z.pkl'); z.columns=NAMES; z.index.name='variable'; z.to_csv(OUT+'perfil_z.csv',float_format='%.4f')
cd=pd.read_pickle('work/ct_director.pkl'); cd.columns=NAMES; cd.index=['Director %d'%i for i in cd.index]; cd.index.name='director'; cd.to_csv(OUT+'ocupacion_director.csv',float_format='%.3f')
cc=pd.read_pickle('work/ct_comp.pkl'); cc.columns=NAMES; cc.index=[{'Bartok':'Bartók','Dvorak':'Dvořák','Mozart':'Mozart'}[i] for i in cc.index]; cc.index.name='obra'; cc.to_csv(OUT+'ocupacion_obra.csv',float_format='%.3f')
tp=pd.read_pickle('work/trans_p.pkl'); tp.index=NAMES; tp.columns=NAMES; tp.to_csv(OUT+'transiciones.csv',float_format='%.4f')
rs=pd.read_pickle('work/runs_sm.pkl'); ep=rs.groupby('c').dur.agg(['count','median','mean']).reset_index(); ep['cluster']=NAMES; ep.drop(columns='c').to_csv(OUT+'episodios_suavizados.csv',index=False,float_format='%.4f')
# --- tomas
tk=pd.read_pickle('work/tomas.pkl'); ft=pd.read_pickle('work/features_all.pkl'); g=ft.groupby('recording_id').agg(frames=('frame','size'),pct_validos=('valido','mean')).reset_index()
occ=pd.crosstab(D.recording_id,D.cluster,normalize='index'); occ.columns=['pct_'+n.split()[0] for n in NAMES]
tt=tk.merge(g,on='recording_id').merge(occ.reset_index(),on='recording_id'); tt['pct_validos']*=100
for c in occ.columns: tt[c]*=100
tt['compositor']=tt.compositor.map({'Bartok':'Bartók','Dvorak':'Dvořák','Mozart':'Mozart'}); tt.rename(columns={'recording_id':'toma','compositor':'obra','vm_med':'vel_media_manos_mediana','vm_p95':'vel_media_manos_p95','asim':'asimetria_mediana','n':'frames_validos','dur':'duracion_s'}).drop(columns=['duracion_s']).to_csv(OUT+'tomas_resumen.csv',index=False,float_format='%.3f')
fu=json.load(open('work/fusion.json')); vv=pd.read_pickle('work/features_all_variantes.pkl').groupby('recording_id').valido.mean(); vb=ft.groupby('recording_id').valido.mean()
pd.DataFrame([dict(toma=k,variante_1=vv[v['variantes'][0]]*100,variante_2=vv[v['variantes'][1]]*100,fusionada=vb[k]*100,marcador_frames_completados=sum(v['frames_completados'].values())) for k,v in fu.items()]).to_csv(OUT+'fusion_variantes.csv',index=False,float_format='%.2f')
occm=pd.read_pickle('work/occ.pkl').drop(columns=['n','nan']).mean().rename('fraccion_ocluida').reset_index().rename(columns={'index':'marcador'}); occm.to_csv(OUT+'oclusion_por_marcador.csv',index=False,float_format='%.4f')
# --- reporte de limpieza compacto
ci=json.load(open('work/clean_info.json')); rep={}
for r,d in ci.items(): rep[r]={m:{k:v for k,v in d[m].items() if k!='longitudes'} for m in d}
json.dump(rep,open(OUT+'reporte_limpieza_54_tomas.json','w'),ensure_ascii=False)
# --- muestra con PCA / t-SNE
fit=(D.frame%4==0).values; sc=StandardScaler().fit(D.loc[fit,FEATS_MEJ].values); pca=PCA().fit(sc.transform(D.loc[fit,FEATS_MEJ].values))
sub=np.load('work/sub_idx.npy'); ts=np.load('work/tsne_sub.npy'); rng=np.random.default_rng(3)
rest=np.setdiff1d(np.arange(len(D)),sub); extra=rng.choice(rest,14000,replace=False); idx=np.concatenate([sub,extra])
S=D.iloc[idx][['recording_id','compositor','director','tiempo_seg','cluster','cluster_suav']+FEATS_MEJ+['batuta_velocidad']].copy().reset_index(drop=True)
P=pca.transform(sc.transform(S[FEATS_MEJ].values))[:,:3]; S['pc1'],S['pc2'],S['pc3']=P.T
S['tsne1']=np.nan; S['tsne2']=np.nan; S.loc[:len(sub)-1,'tsne1']=ts[:,0]; S.loc[:len(sub)-1,'tsne2']=ts[:,1]
S['compositor']=S.compositor.map({'Bartok':'Bartók','Dvorak':'Dvořák','Mozart':'Mozart'}); S['director']=S.director.map(lambda i:'Director %d'%i)
S['cluster_nombre']=S.cluster.map(lambda i:NAMES[i-1]); S.rename(columns={'compositor':'obra'},inplace=True)
S.to_csv(OUT+'muestra_fotogramas.csv.gz',index=False,float_format='%.4f',compression='gzip')
# --- series 30 Hz
s=D[fit][['recording_id','tiempo_seg','velocidad_media_manos','mano_izq_velocidad','mano_der_velocidad','cluster','cluster_suav']]
s.to_csv(OUT+'series_30hz.csv.gz',index=False,float_format='%.3f',compression='gzip')
# --- resumen
res=dict(archivos=62,tomas_unicas=54,pares_fusionados=8,frames_originales=976408,frames_tomas=int(len(ft)),frames_validos=int(len(D)),pct_validos=float(ft.valido.mean()),horas=float(len(ft)/120/3600),fps=120,variables=26,
 k_final=5,silhouette_k5=float(m[(m.conjunto=='MEJ')&(m.algoritmo=='KMeans')&(m.k==5)].silhouette.iloc[0]),db_k5=float(m[(m.conjunto=='MEJ')&(m.algoritmo=='KMeans')&(m.k==5)].davies_bouldin.iloc[0]),
 ari_k5=float(pd.read_pickle('work/estabilidad.pkl').query('k==5').ari.mean()),silueta_media_muestra=json.load(open('work/sil_k5.json'))['media'],
 nmi_director_k5=float(NMI(D.director,D.lab_MEJ_5)),nmi_obra_k5=float(NMI(D.compositor,D.lab_MEJ_5)),dwell=rc['dwell'],kw=rc['kw'],eta2=rc['eta2'],batuta=rc['batuta'],gaps=rc['gaps'],wrist_gaps=rc['wrist_gaps'],pca_var=json.load(open('work/pca_var.json')),energia=json.load(open('work/energy_corr.json')),nombres_clusters=NAMES)
json.dump(res,open(OUT+'resumen_proyecto.json','w'),ensure_ascii=False,indent=1)
print({f:os.path.getsize(OUT+f)//1024 for f in os.listdir(OUT)})
