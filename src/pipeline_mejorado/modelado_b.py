import numpy as np, pandas as pd, json, time, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI, silhouette_samples
from scipy.ndimage import uniform_filter1d
from mocap_pipeline import FEATS_BASE, FEATS_MEJ
warnings.filterwarnings('ignore'); t0=time.time()
F=pd.read_pickle('work/features_sel.pkl'); D=F[F.valido].reset_index(drop=True)
fit=(D.frame%4==0).values
out={}
def orden(lab,k,ref):
    m=[ref[lab==j].mean() for j in range(k)]; o=np.argsort(m); mp=np.empty(k,int); mp[o]=np.arange(k); return mp[lab]
sc={}; Zall={}
for nm,fe in [('MEJ',FEATS_MEJ),('BASE',FEATS_BASE)]:
    s=StandardScaler().fit(D.loc[fit,fe].values); sc[nm]=s; Zall[nm]=s.transform(D[fe].values).astype(np.float32)
ref=D.velocidad_media_manos.values
models={}
for nm,k in [('MEJ',2),('MEJ',3),('MEJ',4),('MEJ',5),('BASE',2),('BASE',3),('BASE',4)]:
    km=KMeans(k,n_init=20,random_state=42).fit(Zall[nm][fit]); lab=orden(km.predict(Zall[nm]),k,ref)
    D[f'lab_{nm}_{k}']=lab.astype(np.int8); models[(nm,k)]=km
    print(nm,k,np.bincount(lab),round(time.time()-t0),flush=True)
D.to_pickle('work/D_labels.pkl')
# --- Estabilidad split-half por tomas
takes=D.recording_id.unique(); rng=np.random.default_rng(7); Z=Zall['MEJ']
ev=rng.choice(len(Z),20000,replace=False); stab=[]
for k in range(2,7):
    for rep in range(10):
        p=rng.permutation(takes); A,B=set(p[:27]),set(p[27:])
        ia=fit&D.recording_id.isin(A).values; ib=fit&D.recording_id.isin(B).values
        la=KMeans(k,n_init=3,random_state=rep).fit(Z[ia]).predict(Z[ev]); lb=KMeans(k,n_init=3,random_state=100+rep).fit(Z[ib]).predict(Z[ev])
        stab.append(dict(k=k,rep=rep,ari=ARI(la,lb)))
    print('stab',k,np.mean([s['ari'] for s in stab if s['k']==k]),round(time.time()-t0),flush=True)
pd.DataFrame(stab).to_pickle('work/estabilidad.pkl')
# --- PCA
pca=PCA().fit(Z[fit]); out['pca_var']=pca.explained_variance_ratio_.tolist()
np.save('work/pca_components.npy',pca.components_[:4])
sub=rng.choice(len(Z),6000,replace=False); P2=pca.transform(Z[sub])[:,:3]
np.save('work/pca_sub.npy',P2); np.save('work/sub_idx.npy',sub)
ts=TSNE(2,perplexity=40,random_state=42,init='pca',max_iter=750).fit_transform(Z[sub]); np.save('work/tsne_sub.npy',ts)
print('tsne',round(time.time()-t0),flush=True)
# silhouette por muestra k=4
ss=silhouette_samples(Z[sub],D['lab_MEJ_4'].values[sub]); np.save('work/sil_samples_k4.npy',ss)
# --- NMI/ARI
nm_rows=[]
for k in [2,3,4]:
    for nm in ['BASE','MEJ']:
        l=D[f'lab_{nm}_{k}']
        nm_rows.append(dict(k=k,conjunto=nm,nmi_director=NMI(D.director,l),nmi_compositor=NMI(D.compositor,l),nmi_toma=NMI(D.recording_id,l)))
    nm_rows.append(dict(k=k,conjunto='ARI BASE-MEJ',nmi_director=ARI(D[f'lab_BASE_{k}'],D[f'lab_MEJ_{k}']),nmi_compositor=np.nan,nmi_toma=np.nan))
pd.DataFrame(nm_rows).to_pickle('work/nmi.pkl'); print(pd.DataFrame(nm_rows).round(3))
print('fin',time.time()-t0)
