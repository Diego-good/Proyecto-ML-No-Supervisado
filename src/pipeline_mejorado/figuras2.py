import numpy as np, pandas as pd, json, warnings, matplotlib, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mp
from matplotlib.colors import ListedColormap
from scipy import stats
from mocap_pipeline import *
exec(open('figuras1.py').read().split("# ---------- Fig 1")[0].split("warnings.filterwarnings('ignore')")[1].split("sel=json.load")[0])  # estilo, save, raw, M
warnings.filterwarnings('ignore')
NAMES=json.load(open('work/names.json')) if __import__('os').path.exists('work/names.json') else ['C1','C2','C3','C4','C5']
DIRC=['#2E6F9E','#E08E2B','#4C9F70','#B6465F','#7C5BA6','#8C6D31']
D=pd.read_pickle('work/D_labels.pkl'); F=pd.read_pickle('work/features_sel.pkl'); tk=pd.read_pickle('work/tomas.pkl'); met=pd.read_pickle('work/metricas_k.pkl')
sel=json.load(open('work/seleccion.json')); fusion=json.load(open('work/fusion.json')); FAv=pd.read_pickle('work/features_all_variantes.pkl')
LAB='lab_MEJ_5'; K=5; D['cl']=D[LAB]; rng=np.random.default_rng(1)
which=sys.argv[1:] 
def want(n): return (not which) or (n in which)
# ---------- Fig 8: ejemplo de limpieza
if want('8'):
    g=[]
    for r in ['Bartok_Conductor03_trial01','Mozart_Conductor03_trial02','Bartok_Conductor01_trial01','Dvorak_Conductor01_trial02']:
        X=raw(r); c=[ci['RWRA_x'],ci['RWRA_y'],ci['RWRA_z']]; z=np.abs(X[:,c]).sum(1)==0
        s,e=nan_runs(z); g=[(i,j) for i,j in zip(s,e) if 4<=j-i<=12 and i>400 and j<len(X)-400]
        if g: break
    i0=g[0][0]
    w0=max(0,i0-300); w1=i0+300; cl,_=clean_marker(X[:,c]); t=np.arange(len(X))/FPS
    fig,axs=plt.subplots(3,1,figsize=(6.8,5.4))
    axs[0].plot(t[w0:w1],X[w0:w1,c[2]],color='#B6465F',lw=.9); axs[0].set_title('(a) Señal cruda de RWRA (coordenada z): las oclusiones aparecen como ceros'); axs[0].set_ylabel('z (m)')
    axs[1].plot(t[w0:w1],np.where(z,np.nan,X[:,c[2]])[w0:w1],color='#aaa',lw=.9,label='datos válidos'); axs[1].plot(t[w0:w1],cl[w0:w1,2],color='#2E6F9E',lw=1.2,label='interpolado ≤12 frames + SG')
    axs[1].set_title('(b) Tras convertir ceros en NaN, interpolar huecos cortos y suavizar'); axs[1].set_ylabel('z (m)'); axs[1].legend(fontsize=7)
    j=i0+g[0][1]-g[0][0]+40; zz=slice(j-60,j+60); axs[2].plot(t[zz],X[zz,c[2]],'o-',ms=2.5,color='#aaa',lw=.6,label='crudo'); axs[2].plot(t[zz],cl[zz,2],color='#2E6F9E',lw=1.4,label='Savitzky-Golay (9 frames, grado 2)')
    axs[2].set_title('(c) Detalle de 1 s: oclusión de ≈0,1 s interpolada y señal suavizada con Savitzky-Golay'); axs[2].set_ylabel('z (m)'); axs[2].set_xlabel('tiempo (s)'); axs[2].legend(fontsize=7)
    fig.tight_layout(); save(fig,'fig08_limpieza')
# ---------- Fig 9: relleno original vs mejorado
if want('9'):
    best=None
    for r in sel['seleccionadas']:
        ci_=json.load(open('work/clean_info.json'))
    import pandas as pd
    cand=[]
    for r in ['Dvorak_Conductor04_trial01','Dvorak_Conductor03_trial02','Bartok_Conductor06_trial03','Dvorak_Conductor01_trial01','Mozart_Conductor03_trial01']:
        if r in fusion: Xr,_=merge_variants(raw(fusion[r]['variantes'][0]),raw(fusion[r]['variantes'][1]),ci)
        else: Xr=raw(r)
        for m in ['LWRA','RWRA','LELB']:
            c=[ci[f'{m}_x'],ci[f'{m}_y'],ci[f'{m}_z']]; z=np.abs(Xr[:,c]).sum(1)==0; s,e=nan_runs(z)
            for i,j in zip(s,e):
                if 80<=j-i<=600 and i>200 and j<len(Xr)-200: cand.append((j-i,r,m,i,j,Xr))
    L,r,m,i,j,Xr=sorted(cand,key=lambda x:abs(x[0]-250))[0]
    c=[ci[f'{m}_x'],ci[f'{m}_y'],ci[f'{m}_z']]; a=Xr[:,c].copy(); a[np.abs(a).sum(1)==0]=np.nan
    orig=pd.DataFrame(a).interpolate(method='linear',limit=12,limit_direction='both').ffill().bfill().values
    orig_sm=savgol_filter(orig,9,2,axis=0); new,_=clean_marker(Xr[:,c])
    sp=lambda p:np.linalg.norm(np.gradient(p,1/FPS,axis=0),axis=1)
    w0,w1=i-240,j+240; t=np.arange(len(Xr))/FPS
    fig,axs=plt.subplots(2,1,figsize=(6.8,4.4),sharex=True)
    axs[0].plot(t[w0:w1],orig_sm[w0:w1,2],color='#B6465F',lw=1.4,label='original: interpolación limitada + relleno por vecino'); axs[0].plot(t[w0:w1],new[w0:w1,2],color='#2E6F9E',lw=1.4,label='mejorado: huecos largos se conservan como NaN')
    axs[0].axvspan(t[i],t[j],color='#eee',zorder=0); axs[0].set_ylabel(f'{m} z (m)'); axs[0].legend(fontsize=7,loc='best'); axs[0].set_title(f'Efecto del relleno en un hueco de {L} frames ({L/FPS:.1f} s) — {r.replace("_"," ")}')
    axs[1].plot(t[w0:w1],sp(orig_sm)[w0:w1],color='#B6465F',lw=1.2); axs[1].plot(t[w0:w1],sp(new)[w0:w1],color='#2E6F9E',lw=1.2); axs[1].axvspan(t[i],t[j],color='#eee',zorder=0)
    axs[1].set_ylabel('velocidad (m/s)'); axs[1].set_xlabel('tiempo (s)'); axs[1].text(t[(i+j)//2],axs[1].get_ylim()[1]*.8,'velocidad ≈ 0 artificial',ha='center',color='#B6465F',fontsize=7.5)
    fig.tight_layout(); save(fig,'fig09_relleno')
# ---------- Fig 10: retención
if want('10'):
    vb=FA=pd.read_pickle('work/features_all.pkl').groupby('recording_id').valido.mean(); dr=tk.set_index('recording_id').director
    order=vb.sort_values().index; vv=FAv.groupby('recording_id').valido.mean()
    fig,ax=plt.subplots(figsize=(6.8,3.3)); ax.bar(range(len(order)),[vb[r]*100 for r in order],color=[DIRC[dr[r]-1] for r in order])
    for x,r in enumerate(order):
        if r in fusion: ax.plot(x,max(vv[v] for v in fusion[r]['variantes'])*100,marker='v',color='k',ms=5,ls='')
    ax.set_xticks(range(len(order))); ax.set_xticklabels([r.replace('_trial','·t').replace('Conductor','D') for r in order],rotation=90,fontsize=5.2); ax.set_ylabel('% de fotogramas válidos'); ax.set_ylim(0,105)
    hs=[mp.Patch(color=DIRC[i],label=f'Director {i+1}') for i in range(6)]+[plt.Line2D([],[],marker='v',color='k',ls='',label='mejor variante antes de fusionar')]
    ax.legend(handles=hs,fontsize=6.3,ncol=4,loc='upper left'); ax.set_title('Fotogramas con todas las variables válidas, por toma'); fig.tight_layout(); save(fig,'fig10_retencion')
# ---------- Fig 11: sensibilidad
if want('11'):
    s=pd.read_pickle('work/sens.pkl'); a=s[s.ventana==9]; b=s[s.max_hueco==12].sort_values('ventana')
    fig,axs=plt.subplots(1,2,figsize=(6.8,2.7)); axs[0].plot(a.max_hueco,a.pct_validos*100,'o-',color='#2E6F9E'); axs[0].set_xlabel('longitud máxima de hueco interpolado (frames)'); axs[0].set_ylabel('% de fotogramas válidos'); axs[0].axvline(12,color='#B22',ls='--',lw=.8); axs[0].set_title('Huecos: retención de datos')
    axs[1].plot(b.ventana,b.acel_p95,'o-',color='#B6465F',label='aceleración p95 (m/s²), eje izq.'); axs[1].set_xlabel('ventana Savitzky-Golay (frames)'); axs[1].set_ylabel('aceleración p95 (m/s²)'); axs[1].set_xticks([5,9,15]); axs[1].axvline(9,color='#B22',ls='--',lw=.8)
    ax2=axs[1].twinx(); ax2.plot(b.ventana,b.vel_p95,'s-',color='#2E6F9E'); ax2.set_ylabel('velocidad p95 (m/s)'); ax2.grid(False); ax2.spines['right'].set_visible(True); axs[1].set_title('Ventana: ruido en la aceleración')
    fig.tight_layout(); save(fig,'fig11_sensibilidad')
# ---------- Fig 12: distribuciones de velocidad
if want('12'):
    fig,axs=plt.subplots(1,2,figsize=(6.8,2.7)); bins=np.linspace(0,3.5,70)
    for nm,col,cc in [('Mano izq.','mano_izq_velocidad','#2E6F9E'),('Mano der.','mano_der_velocidad','#B6465F'),('Codo izq.','codo_izq_velocidad','#7aa6c9'),('Codo der.','codo_der_velocidad','#d98e9c')]:
        axs[0].hist(D[col].clip(upper=3.5),bins=bins,density=True,histtype='step',color=cc,lw=1.3,label=nm)
    axs[0].set_xlabel('velocidad (m/s)'); axs[0].set_ylabel('densidad'); axs[0].legend(fontsize=7); axs[0].set_title('Velocidad de manos y codos')
    axs[1].hist(D.velocidad_media_manos.clip(upper=3),bins=70,density=True,color='#4C9F70',alpha=.8); axs[1].axvline(D.velocidad_media_manos.median(),color='k',ls='--',lw=.8); axs[1].set_xlabel('velocidad media de las manos (m/s)'); axs[1].set_title('Intensidad global del gesto')
    axs[1].text(D.velocidad_media_manos.median()+.05,axs[1].get_ylim()[1]*.85,f'mediana = {D.velocidad_media_manos.median():.2f} m/s',fontsize=7.5)
    fig.tight_layout(); save(fig,'fig12_velocidades')
# ---------- Fig 13 y 14: energía vs velocidad, correlaciones
if want('13'):
    s=D.sample(60000,random_state=1); fig,axs=plt.subplots(1,2,figsize=(6.8,2.8))
    axs[0].hexbin(s.mano_der_velocidad,s.mano_der_energia_rel,gridsize=45,bins='log',cmap='viridis',extent=(0,3,0,4.5)); axs[0].set_xlabel('velocidad mano derecha (m/s)'); axs[0].set_ylabel('energía cinética relativa ½v²'); axs[0].set_title('Energía vs. velocidad')
    rho=stats.spearmanr(D.mano_der_velocidad,D.mano_der_energia_rel)[0]; pr=np.corrcoef(D.mano_der_velocidad,D.mano_der_energia_rel)[0,1]
    axs[0].text(.1,3.8,f'Spearman = {rho:.2f}\nPearson = {pr:.2f}',fontsize=7.5,color='w')
    cmat=D[FEATS_BASE].corr(method='spearman'); axs[1].imshow(cmat.values,cmap='RdBu_r',vmin=-1,vmax=1); axs[1].grid(False); axs[1].set_xticks([]); axs[1].set_yticks([]); axs[1].set_title('Correlación (Spearman), variables BASE')
    fig.tight_layout(); save(fig,'fig13_energia'); json.dump({'rho':float(rho),'pearson':float(pr)},open('work/energy_corr.json','w'))
if want('14'):
    cm=D[FEATS_MEJ].corr(method='spearman'); fig,ax=plt.subplots(figsize=(6.6,5.6)); im=ax.imshow(cm.values,cmap='RdBu_r',vmin=-1,vmax=1); ax.grid(False)
    lab=[c.replace('mano_','M ').replace('codo_','C ').replace('izq','I').replace('der','D').replace('velocidad','vel').replace('aceleracion','ac').replace('asimetria','asim').replace('angulo','áng').replace('_',' ') for c in FEATS_MEJ]
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab,rotation=90,fontsize=6); ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab,fontsize=6); fig.colorbar(im,ax=ax,fraction=.04,pad=.02,label='correlación de Spearman'); ax.set_title('Correlación entre las 26 variables del conjunto mejorado')
    save(fig,'fig14_correlacion')
    cm_=cm.where(~np.eye(len(cm),dtype=bool)).abs().stack().sort_values(ascending=False); json.dump({'pares_alta_corr':[(a,b,float(v)) for (a,b),v in cm_.iloc[:12:2].items()],'n_mayor_07':int((cm_>0.7).sum()//2)},open('work/corr_top.json','w'))
# ---------- Fig 15: PCA
if want('15'):
    pv=np.array(json.load(open('work/pca_var.json'))); comp=np.load('work/pca_components.npy')
    fig,axs=plt.subplots(1,2,figsize=(6.8,3.0),gridspec_kw={'width_ratios':[1,1.5]}); axs[0].bar(range(1,len(pv)+1),pv*100,color='#2E6F9E'); axs[0].plot(range(1,len(pv)+1),np.cumsum(pv)*100,'o-',color='#B6465F',ms=3); axs[0].set_xlabel('componente'); axs[0].set_ylabel('% de varianza'); axs[0].set_title('Varianza explicada'); axs[0].axhline(80,color='#888',ls=':',lw=.8)
    lab=[c.replace('mano_','M ').replace('codo_','C ').replace('izq','I').replace('der','D').replace('velocidad','vel').replace('aceleracion','ac').replace('asimetria','asim').replace('angulo','áng').replace('_',' ') for c in FEATS_MEJ]
    im=axs[1].imshow(comp[:3].T,aspect='auto',cmap='RdBu_r',vmin=-.5,vmax=.5); axs[1].grid(False); axs[1].set_xticks(range(3)); axs[1].set_xticklabels(['PC1','PC2','PC3']); axs[1].set_yticks(range(len(lab))); axs[1].set_yticklabels(lab,fontsize=5.5); axs[1].set_title('Cargas de las tres primeras componentes'); fig.colorbar(im,ax=axs[1],fraction=.05,pad=.02)
    fig.tight_layout(); save(fig,'fig15_pca')
# ---------- Fig 16-18: selección de k
if want('16'):
    m=met[met.conjunto=='MEJ']; fig,axs=plt.subplots(2,2,figsize=(6.8,4.8))
    km=m[m.algoritmo=='KMeans']; wd=m[m.algoritmo=='Ward_aprox']
    axs[0,0].plot(km.k,km.inercia/1e6,'o-',color='#2E6F9E'); axs[0,0].set_title('Inercia (método del codo)'); axs[0,0].set_ylabel('inercia ×10⁶')
    for ax,col,tt in [(axs[0,1],'silhouette','Silhouette (mayor es mejor)'),(axs[1,0],'davies_bouldin','Davies-Bouldin (menor es mejor)'),(axs[1,1],'calinski_harabasz','Calinski-Harabasz (mayor es mejor)')]:
        ax.plot(km.k,km[col],'o-',color='#2E6F9E',label='K-Means'); ax.plot(wd.k,wd[col],'s--',color='#E08E2B',label='Ward aprox.'); ax.set_title(tt)
    axs[0,1].legend(fontsize=7)
    for ax in axs.flat: ax.set_xlabel('k')
    for ax in axs.flat: ax.axvline(5,color='#B22',ls=':',lw=.9)
    fig.tight_layout(); save(fig,'fig16_seleccion_k')
if want('17'):
    st=pd.read_pickle('work/estabilidad.pkl'); gm=pd.read_pickle('work/gmm.pkl'); fig,axs=plt.subplots(1,2,figsize=(6.8,2.8))
    axs[0].boxplot([st[st.k==k].ari for k in range(2,7)],tick_labels=range(2,7),patch_artist=True,boxprops=dict(facecolor='#cfe0ee'),medianprops=dict(color='#B22')); axs[0].set_xlabel('k'); axs[0].set_ylabel('ARI entre mitades de tomas'); axs[0].set_title('Estabilidad (10 particiones al azar)')
    axs[1].plot(gm.k,gm.bic/1e5,'o-',color='#4C9F70',label='BIC'); axs[1].plot(gm.k,gm.aic/1e5,'s--',color='#7C5BA6',label='AIC'); axs[1].set_xlabel('componentes de la mezcla'); axs[1].set_ylabel('criterio ×10⁵'); axs[1].set_title('Mezclas gaussianas (cov. completa)'); axs[1].legend(fontsize=7)
    fig.tight_layout(); save(fig,'fig17_estabilidad_gmm')
if want('18'):
    fig,axs=plt.subplots(1,2,figsize=(6.8,2.7)); cs={'BASE':'#B6465F','MEJ':'#2E6F9E','VENT':'#4C9F70'}; nm={'BASE':'BASE (posiciones absolutas)','MEJ':'MEJ (marco corporal)','VENT':'VENT (MEJ + ventana 0,5 s)'}
    for n in ['BASE','MEJ','VENT']:
        g=met[(met.conjunto==n)&(met.algoritmo=='KMeans')]; axs[0].plot(g.k,g.silhouette,'o-',color=cs[n],label=nm[n]); axs[1].plot(g.k,g.davies_bouldin,'o-',color=cs[n])
    axs[0].set_title('Silhouette (K-Means)'); axs[1].set_title('Davies-Bouldin (K-Means)'); axs[0].legend(fontsize=6.5)
    for a in axs: a.set_xlabel('k')
    fig.tight_layout(); save(fig,'fig18_conjuntos')
# ---------- Fig 19-21: proyecciones
sub=np.load('work/sub_idx.npy'); P2=np.load('work/pca_sub.npy'); ts=np.load('work/tsne_sub.npy'); cl=D[LAB].values[sub]; pv=json.load(open('work/pca_var.json'))
if want('19'):
    fig,ax=plt.subplots(figsize=(5.2,4.2))
    for j in range(K): ax.scatter(P2[cl==j,0],P2[cl==j,1],s=4,color=CL[j],alpha=.55,label=NAMES[j])
    ax.set_xlabel(f'PC1 ({pv[0]*100:.1f} %)'); ax.set_ylabel(f'PC2 ({pv[1]*100:.1f} %)'); ax.legend(markerscale=3,fontsize=7); ax.set_title('Proyección PCA de los 5 clusters (6 000 fotogramas)'); fig.tight_layout(); save(fig,'fig19_pca_clusters')
if want('20'):
    fig,ax=plt.subplots(figsize=(5.2,4.2))
    for j in range(K): ax.scatter(ts[cl==j,0],ts[cl==j,1],s=4,color=CL[j],alpha=.6,label=NAMES[j])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2'); ax.legend(markerscale=3,fontsize=7); ax.set_title('Proyección t-SNE (perplejidad 40)'); fig.tight_layout(); save(fig,'fig20_tsne')
if want('21'):
    ss=np.load('work/sil_samples_k4.npy'); 
    from sklearn.metrics import silhouette_samples
    from sklearn.preprocessing import StandardScaler
    Z=StandardScaler().fit(D.loc[D.frame%4==0,FEATS_MEJ].values).transform(D[FEATS_MEJ].values)[sub]; ss=silhouette_samples(Z,cl); np.save('work/sil_samples_k5.npy',ss)
    fig,ax=plt.subplots(figsize=(5.2,3.6)); y=0
    for j in range(K):
        v=np.sort(ss[cl==j]); ax.fill_betweenx(np.arange(y,y+len(v)),0,v,color=CL[j],lw=0); ax.text(-.38,y+len(v)/2,NAMES[j].split()[0],fontsize=7.5,va='center'); y+=len(v)+80
    ax.axvline(ss.mean(),color='k',ls='--',lw=.8); ax.set_xlabel('coeficiente de silueta'); ax.set_yticks([]); ax.set_title(f'Silueta por cluster (media = {ss.mean():.3f})'); ax.set_xlim(-.4,.7); fig.tight_layout(); save(fig,'fig21_silueta')
    json.dump({'media':float(ss.mean()),'neg_pct':float((ss<0).mean()),'por_cluster':[float(ss[cl==j].mean()) for j in range(K)],'neg_por_cluster':[float((ss[cl==j]<0).mean()) for j in range(K)]},open('work/sil_k5.json','w'))
print('batch1 done')
