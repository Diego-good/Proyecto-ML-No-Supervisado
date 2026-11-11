import numpy as np, pandas as pd, json, time, warnings
from scipy.ndimage import uniform_filter1d
from scipy import stats
from mocap_pipeline import *
warnings.filterwarnings('ignore'); t0=time.time()
D=pd.read_pickle('work/D_labels.pkl'); sel=json.load(open('work/seleccion.json'))['seleccionadas']
R={}
# ---- dwell / smoothing / transitions (k=5)
def segments(g):
    fr=g.frame.values; br=np.flatnonzero(np.diff(fr)!=1)+1
    return np.split(np.arange(len(g)),br)
def smooth(lab,k,w=61):
    oh=np.eye(k)[lab]; return uniform_filter1d(oh,size=w,axis=0,mode='nearest').argmax(1)
K=5; runs_raw=[]; runs_sm=[]; trans=np.zeros((K,K)); sm_all=np.empty(len(D),np.int8); changes_raw=0; changes_sm=0; nfr=0
for r,g in D.groupby('recording_id',sort=False):
    ix=g.index.values; lab=g['lab_MEJ_5'].values
    for seg in segments(g):
        l=lab[seg]; s=smooth(l,K); sm_all[ix[seg]]=s
        for arr,store in [(l,runs_raw),(s,runs_sm)]:
            ch=np.flatnonzero(np.diff(arr)!=0)+1; st=np.concatenate([[0],ch]); en=np.concatenate([ch,[len(arr)]])
            for a,b in zip(st,en): store.append((arr[a],(b-a)/FPS))
        changes_raw+=int((np.diff(l)!=0).sum()); changes_sm+=int((np.diff(s)!=0).sum()); nfr+=len(l)
        ch=np.flatnonzero(np.diff(s)!=0)
        for a in ch: trans[s[a],s[a+1]]+=1
D['lab_MEJ_5_suav']=sm_all; D.to_pickle('work/D_labels.pkl')
rr=pd.DataFrame(runs_raw,columns=['c','dur']); rs=pd.DataFrame(runs_sm,columns=['c','dur'])
R['dwell']={'raw_median':float(rr.dur.median()),'raw_mean':float(rr.dur.mean()),'sm_median':float(rs.dur.median()),'sm_mean':float(rs.dur.mean()),
            'n_raw':len(rr),'n_sm':len(rs),'cambios_por_s_raw':changes_raw/(nfr/FPS),'cambios_por_s_sm':changes_sm/(nfr/FPS),
            'por_cluster_sm_median':rs.groupby('c').dur.median().round(3).tolist(),'por_cluster_sm_mean':rs.groupby('c').dur.mean().round(3).tolist(),
            'por_cluster_raw_median':rr.groupby('c').dur.median().round(3).tolist()}
rr.to_pickle('work/runs_raw.pkl'); rs.to_pickle('work/runs_sm.pkl'); np.save('work/trans.npy',trans)
print(R['dwell'])
# ---- descriptivos por toma / director / compositor
tk=D.groupby(['recording_id','compositor','director']).agg(vm_med=('velocidad_media_manos','median'),vm_p95=('velocidad_media_manos',lambda x:np.percentile(x,95)),
     asim=('asimetria_velocidad_manos','median'),n=('frame','size'),dur=('tiempo_seg','max')).reset_index()
tk.to_pickle('work/tomas.pkl')
H_comp=stats.kruskal(*[g.vm_med.values for _,g in tk.groupby('compositor')]); H_dir=stats.kruskal(*[g.vm_med.values for _,g in tk.groupby('director')])
R['kw']={'compositor':[float(H_comp.statistic),float(H_comp.pvalue)],'director':[float(H_dir.statistic),float(H_dir.pvalue)]}
R['ntomas']=tk.groupby('compositor').size().to_dict(); R['ntomas_dir']={int(k):int(v) for k,v in tk.groupby('director').size().items()}
# varianza explicada (eta^2 aprox) de la mediana de velocidad por director y compositor
def eta2(col): 
    gm=tk.vm_med.mean(); ss_b=sum(len(g)*(g.vm_med.mean()-gm)**2 for _,g in tk.groupby(col)); return float(ss_b/((tk.vm_med-gm)**2).sum())
R['eta2']={'director':eta2('director'),'compositor':eta2('compositor')}
print(R['kw'],R['eta2'],R['ntomas'])
# ---- batuta
B=D[D.batuta_velocidad.notna()]
R['batuta']={'tomas':int(B.recording_id.nunique()),'directores':sorted(int(x) for x in B.director.unique()),'frames':int(len(B)),
   'ratio_med':float((B.batuta_velocidad/B.mano_der_velocidad.clip(lower=0.02)).median()),'corr_spearman':float(stats.spearmanr(B.batuta_velocidad,B.mano_der_velocidad)[0]),
   'long_med':float(B.batuta_longitud.median()),'vel_med':float(B.batuta_velocidad.median())}
print(R['batuta'])
# ---- huecos largos en muñecas
ci=json.load(open('work/clean_info.json'))
tot=0; larg=0; fr_oc=0
for r in sel:
    for m in ['LWRA','LWRB','RWRA','RWRB']:
        L=ci[r][m]['longitudes']; N=len(D[D.recording_id==r]) if False else None
        larg+=sum(x for x in L if x>12); tot+=sum(L); 
R['wrist_gaps']={'frames_ocluidos':tot,'frames_en_huecos_largos':larg}
allg=[x for r in sel for m in MARKERS for x in ci[r][m]['longitudes']]
R['gaps']={'n':len(allg),'pct_cortos':float(np.mean(np.array(allg)<=12)),'max':int(max(allg)),'mediana':float(np.median(allg))}
print(R['wrist_gaps'],R['gaps'])
# ---- sensibilidad preprocesamiento
cols=json.load(open('work/cols.json')); colidx={c:i for i,c in enumerate(cols)}; M=np.load('work/raw.npy',mmap_mode='r'); rid=np.load('work/rid.npy')
fusion=json.load(open('work/fusion.json'))
def getX(r):
    m=np.flatnonzero(rid==r); X=np.asarray(M[m[0]:m[-1]+1]); return X[np.argsort(X[:,colidx['frame']],kind='stable')]
TAKES={}
for b in sel:
    if b in fusion: TAKES[b]=merge_variants(getX(fusion[b]['variantes'][0]),getX(fusion[b]['variantes'][1]),colidx)[0]
    else: TAKES[b]=getX(b)
rows=[]
for mg,w in [(0,9),(6,9),(12,9),(24,9),(48,9),(12,5),(12,15)]:
    vs=[];hv=[];ac=[];nt=0
    for r in sel:
        X=TAKES[r]
        df,_=process_recording(X,colidx,max_gap=mg,win=w); ok=df[FEATS_BASE+FEATS_MEJ].notna().all(axis=1)
        vs.append(ok.sum()); nt+=len(df); hv.append(df.loc[ok,'velocidad_media_manos'].values); ac.append(df.loc[ok,'mano_der_aceleracion'].values)
    hv=np.concatenate(hv); ac=np.concatenate(ac)
    rows.append(dict(max_hueco=mg,ventana=w,validos=int(sum(vs)),pct_validos=sum(vs)/nt,vel_mediana=float(np.median(hv)),vel_p95=float(np.percentile(hv,95)),acel_mediana=float(np.median(ac)),acel_p95=float(np.percentile(ac,95))))
    print(rows[-1],round(time.time()-t0),flush=True)
pd.DataFrame(rows).to_pickle('work/sens.pkl')
json.dump(R,open('work/res_c.json','w'),indent=1)
print('fin')
