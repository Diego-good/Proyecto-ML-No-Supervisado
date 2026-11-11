import numpy as np, pandas as pd, json, time, re, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from mocap_pipeline import FEATS_BASE, FEATS_MEJ
warnings.filterwarnings('ignore')
t0=time.time()
F=pd.read_pickle('work/features_all.pkl')
# --- deduplicación de variantes _1/_2 (mismas trayectorias) ---
v=F.groupby('recording_id').valido.mean()
groups={}
for r in v.index:
    b=re.sub(r'_[12]$','',r); groups.setdefault(b,[]).append(r)
sel=[max(rs,key=lambda x:v[x]) for b,rs in groups.items()]
json.dump({'seleccionadas':sel,'descartadas':[r for r in v.index if r not in sel]},open('work/seleccion.json','w'),indent=1)
print(len(sel),'tomas seleccionadas')
# ventana temporal (extensión): media y desv. móvil de 0,5 s
def add_window(df):
    out=[]
    for r,g in df.groupby('recording_id',sort=False):
        s=g['velocidad_media_manos']; a=(g['mano_izq_aceleracion']+g['mano_der_aceleracion'])/2
        g=g.copy()
        g['vm_media_0p5s']=s.rolling(61,center=True,min_periods=40).mean().values
        g['vm_desv_0p5s']=s.rolling(61,center=True,min_periods=40).std().values
        g['acel_media_0p5s']=a.rolling(61,center=True,min_periods=40).mean().values
        out.append(g)
    return pd.concat(out)
F=F[F.recording_id.isin(sel)].copy()
F=add_window(F)
FEATS_VENT=FEATS_MEJ+['vm_media_0p5s','vm_desv_0p5s','acel_media_0p5s']
F['valido_v']=F.valido & F[FEATS_VENT].notna().all(axis=1)
F.to_pickle('work/features_sel.pkl')
D=F[F.valido].copy()
print('frames válidos en tomas seleccionadas',len(D))
rng=np.random.default_rng(42)
res=[]; gm=[]
for name,feats,mask in [('BASE',FEATS_BASE,F.valido),('MEJ',FEATS_MEJ,F.valido),('VENT',FEATS_VENT,F.valido_v)]:
    Dm=F[mask]; Df=Dm[Dm.frame%4==0]
    sc=StandardScaler().fit(Df[feats].values); Z=sc.transform(Df[feats].values)
    n=len(Z); ev=rng.choice(n,8000,replace=False); wi=rng.choice(n,4000,replace=False)
    print(name,'ajuste',n,flush=True)
    for k in range(2,9):
        km=KMeans(k,n_init=10,random_state=42).fit(Z); lab=km.labels_
        res.append(dict(conjunto=name,algoritmo='KMeans',k=k,inercia=km.inertia_,
            silhouette=silhouette_score(Z[ev],lab[ev]),davies_bouldin=davies_bouldin_score(Z,lab),
            calinski_harabasz=calinski_harabasz_score(Z,lab),tamanos=np.bincount(lab,minlength=k).tolist()))
        ag=AgglomerativeClustering(k,linkage='ward').fit(Z[wi]); cen=np.vstack([Z[wi][ag.labels_==j].mean(0) for j in range(k)])
        la=((Z[:,None,:]-cen[None])**2).sum(2).argmin(1) if n<1 else np.concatenate([((Z[i:i+20000,None,:]-cen[None])**2).sum(2).argmin(1) for i in range(0,n,20000)])
        res.append(dict(conjunto=name,algoritmo='Ward_aprox',k=k,inercia=np.nan,
            silhouette=silhouette_score(Z[ev],la[ev]),davies_bouldin=davies_bouldin_score(Z,la),
            calinski_harabasz=calinski_harabasz_score(Z,la),tamanos=np.bincount(la,minlength=k).tolist()))
        print(name,k,round(res[-2]['silhouette'],3),round(res[-1]['silhouette'],3),round(time.time()-t0),flush=True)
    if name=='MEJ':
        g40=rng.choice(n,40000,replace=False)
        for k in range(1,9):
            g=GaussianMixture(k,covariance_type='full',n_init=2,random_state=42,reg_covar=1e-4,max_iter=200).fit(Z[g40])
            gm.append(dict(k=k,bic=g.bic(Z[g40]),aic=g.aic(Z[g40])))
            print('gmm',k,round(gm[-1]['bic']),round(time.time()-t0),flush=True)
pd.DataFrame(res).to_pickle('work/metricas_k.pkl'); pd.DataFrame(gm).to_pickle('work/gmm.pkl')
print('fin',time.time()-t0)
