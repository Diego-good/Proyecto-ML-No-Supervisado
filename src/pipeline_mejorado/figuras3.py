import numpy as np, pandas as pd, json, warnings, matplotlib, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mp
from scipy import stats
from mocap_pipeline import *
exec(open('figuras1.py').read().split("# ---------- Fig 1")[0].split("warnings.filterwarnings('ignore')")[1].split("sel=json.load")[0])
warnings.filterwarnings('ignore')
NAMES=json.load(open('work/names.json')); DIRC=['#2E6F9E','#E08E2B','#4C9F70','#B6465F','#7C5BA6','#8C6D31']
D=pd.read_pickle('work/D_labels.pkl'); tk=pd.read_pickle('work/tomas.pkl'); K=5; LAB='lab_MEJ_5'; D['cl']=D[LAB]
which=sys.argv[1:]
def want(n): return (not which) or (n in which)
fit=D.frame%4==0; mu=D.loc[fit,FEATS_MEJ].mean(); sd=D.loc[fit,FEATS_MEJ].std()
short=lambda c:c.replace('mano_','M ').replace('codo_','C ').replace('izq','I').replace('der','D').replace('velocidad','vel').replace('aceleracion','ac').replace('asimetria','asim').replace('angulo','áng').replace('_',' ')
# ---- Fig 22 perfiles z
if want('22'):
    Zp=((D[FEATS_MEJ]-mu)/sd).groupby(D.cl).mean().T; fig,ax=plt.subplots(figsize=(6.0,6.6)); im=ax.imshow(Zp.values,cmap='RdBu_r',vmin=-1.6,vmax=1.6,aspect='auto'); ax.grid(False)
    ax.set_xticks(range(K)); ax.set_xticklabels(NAMES,fontsize=7,rotation=20,ha='right'); ax.set_yticks(range(len(FEATS_MEJ))); ax.set_yticklabels([short(c) for c in FEATS_MEJ],fontsize=7)
    for i in range(Zp.shape[0]):
        for j in range(K): ax.text(j,i,f'{Zp.values[i,j]:.1f}',ha='center',va='center',fontsize=6.3,color='k' if abs(Zp.values[i,j])<1.1 else 'w')
    fig.colorbar(im,ax=ax,fraction=.04,pad=.02,label='media z-score'); ax.set_title('Perfil estandarizado de cada cluster'); fig.tight_layout(); save(fig,'fig22_perfil_z')
    Zp.to_pickle('work/perfil_z.pkl')
# ---- Fig 23 perfiles físicos
if want('23'):
    D['acel_media']=(D.mano_izq_aceleracion+D.mano_der_aceleracion)/2
    fig,axs=plt.subplots(1,3,figsize=(6.9,2.6))
    def bars(ax,cols,labs,title,ylab,cc):
        w=.8/len(cols)
        for i,c in enumerate(cols):
            g=D.groupby('cl')[c]; m=g.median(); q1=g.quantile(.25); q3=g.quantile(.75)
            ax.bar(np.arange(K)+i*w-.4+w/2,m,w,yerr=[m-q1,q3-m],color=cc[i],label=labs[i],error_kw=dict(lw=.7,capsize=1.5)); 
        ax.set_xticks(range(K)); ax.set_xticklabels([n.split()[0] for n in NAMES]); ax.set_title(title); ax.set_ylabel(ylab); ax.legend(fontsize=6.3)
    bars(axs[0],['mano_izq_velocidad','mano_der_velocidad'],['mano izq.','mano der.'],'Velocidad (mediana, IQR)','m/s',['#2E6F9E','#B6465F'])
    bars(axs[1],['mano_izq_aceleracion','mano_der_aceleracion'],['mano izq.','mano der.'],'Aceleración (mediana, IQR)','m/s²',['#2E6F9E','#B6465F'])
    bars(axs[2],['angulo_codo_izq','angulo_codo_der'],['codo izq.','codo der.'],'Ángulo del codo (mediana, IQR)','grados',['#2E6F9E','#B6465F'])
    fig.tight_layout(); save(fig,'fig23_perfil_fisico')
    tab=D.groupby('cl').agg(vel=('velocidad_media_manos','median'),vel_izq=('mano_izq_velocidad','median'),vel_der=('mano_der_velocidad','median'),ac_izq=('mano_izq_aceleracion','median'),ac_der=('mano_der_aceleracion','median'),asim=('asimetria_velocidad_manos','median'),
        ang_i=('angulo_codo_izq','median'),ang_d=('angulo_codo_der','median'),dist_i=('mano_izq_dist_torso','median'),dist_d=('mano_der_dist_torso','median'),fre_i=('mano_izq_fre','median'),fre_d=('mano_der_fre','median'),ver_i=('mano_izq_ver','median'),ver_d=('mano_der_ver','median'),lat_i=('mano_izq_lat','median'),lat_d=('mano_der_lat','median'),n=('frame','size'))
    tab['pct']=tab.n/tab.n.sum()*100; tab.to_pickle('work/tabla_clusters.pkl'); print(tab.round(2).T.to_string())
# ---- Fig 24 densidades de posición
if want('24'):
    fig,axs=plt.subplots(2,K,figsize=(6.9,3.4),sharex=True,sharey=True)
    for r,(side,nm) in enumerate([('mano_izq','mano izquierda'),('mano_der','mano derecha')]):
        for j in range(K):
            s=D[D.cl==j].sample(min(25000,(D.cl==j).sum()),random_state=1); axs[r,j].hexbin(s[f'{side}_lat'],s[f'{side}_ver'],gridsize=24,extent=(-.9,.9,-.7,.5),bins='log',cmap='Blues',mincnt=1)
            axs[r,j].plot(0,0,'k+',ms=5); axs[r,j].grid(False)
            if r==0: axs[r,j].set_title(NAMES[j].split()[0]+'\n'+' '.join(NAMES[j].split()[1:]),fontsize=7)
            if j==0: axs[r,j].set_ylabel(nm+'\nvertical (m)',fontsize=7)
            if r==1: axs[r,j].set_xlabel('lateral (m)',fontsize=7)
    fig.tight_layout(); save(fig,'fig24_densidad_posicion')
# ---- Fig 25 tamaños y composición por compositor
if want('25'):
    fig,axs=plt.subplots(1,2,figsize=(6.8,2.9)); pc=D.cl.value_counts(normalize=True).sort_index()*100; axs[0].bar(range(K),pc.values,color=CL); axs[0].set_xticks(range(K)); axs[0].set_xticklabels([n.split()[0] for n in NAMES]); axs[0].set_ylabel('% de fotogramas'); axs[0].set_title('Tamaño de cada cluster')
    for i,v in enumerate(pc.values): axs[0].text(i,v+.5,f'{v:.1f}',ha='center',fontsize=7.5)
    ct=pd.crosstab(D.compositor,D.cl,normalize='index')*100; bot=np.zeros(3)
    for j in range(K): axs[1].barh(['Bartók','Dvořák','Mozart'],ct[j].values,left=bot,color=CL[j],label=NAMES[j].split()[0]); bot+=ct[j].values
    axs[1].set_xlabel('% de fotogramas de la obra'); axs[1].set_title('Composición por obra'); axs[1].legend(fontsize=6.5,ncol=5,loc='lower center',bbox_to_anchor=(.5,-.38)); axs[1].invert_yaxis()
    fig.tight_layout(); save(fig,'fig25_tamano_obra')
# ---- Fig 26 ocupación por director
if want('26'):
    ct=pd.crosstab(D.director,D.cl,normalize='index')*100; fig,ax=plt.subplots(figsize=(6.6,3.0)); bot=np.zeros(6)
    for j in range(K): ax.bar([f'Director {i}' for i in ct.index],ct[j].values,bottom=bot,color=CL[j],label=NAMES[j]); bot+=ct[j].values
    ax.set_ylabel('% de fotogramas'); ax.set_title('Ocupación de los clusters por director'); ax.legend(fontsize=6.5,ncol=5,loc='lower center',bbox_to_anchor=(.5,-.3)); fig.tight_layout(); save(fig,'fig26_director')
    ct.to_pickle('work/ct_director.pkl'); (pd.crosstab(D.compositor,D.cl,normalize='index')*100).to_pickle('work/ct_comp.pkl')
# ---- Fig 27 velocidad por director/compositor
if want('27'):
    fig,axs=plt.subplots(1,2,figsize=(6.8,2.9),gridspec_kw={'width_ratios':[2,1]})
    data=[tk[tk.director==d].vm_med.values for d in range(1,7)]; bp=axs[0].boxplot(data,tick_labels=[f'D{d}' for d in range(1,7)],patch_artist=True,medianprops=dict(color='k'),widths=.55)
    for p,c in zip(bp['boxes'],DIRC): p.set_facecolor(c); p.set_alpha(.55)
    for i,d in enumerate(range(1,7)):
        g=tk[tk.director==d]; axs[0].scatter(np.full(len(g),i+1)+np.random.default_rng(0).uniform(-.12,.12,len(g)),g.vm_med,s=9,c=[COMP[c] for c in g.compositor],zorder=3,edgecolor='k',lw=.3)
    axs[0].set_ylabel('mediana de velocidad media de manos (m/s)'); axs[0].set_title('Por director (cada punto = una toma)')
    data=[tk[tk.compositor==c].vm_med.values for c in ['Bartok','Dvorak','Mozart']]; bp=axs[1].boxplot(data,tick_labels=['Bartók','Dvořák','Mozart'],patch_artist=True,medianprops=dict(color='k'),widths=.55)
    for p,c in zip(bp['boxes'],[COMP['Bartok'],COMP['Dvorak'],COMP['Mozart']]): p.set_facecolor(c); p.set_alpha(.7)
    axs[1].set_title('Por obra'); fig.tight_layout(); save(fig,'fig27_velocidad_director')
# ---- Fig 28 líneas de tiempo
if want('28'):
    from scipy.ndimage import uniform_filter1d
    takes=['Bartok_Conductor04_trial01','Dvorak_Conductor04_trial02','Mozart_Conductor04_trial01']
    fig,axs=plt.subplots(9,1,figsize=(6.8,5.8),gridspec_kw={'height_ratios':[2.2,.6,.6]*3},sharex=True); cmap=matplotlib.colors.ListedColormap(CL)
    for n,r in enumerate(takes):
        g=D[(D.recording_id==r)&(D.tiempo_seg<60)]; ax=axs[3*n]; ax.plot(g.tiempo_seg,g.velocidad_media_manos,color='#444',lw=.7); ax.set_ylabel('m/s',fontsize=7); ax.set_title(r.replace('_',' '),fontsize=8,loc='left',pad=2)
        for k,(col,lab) in enumerate([(LAB,'cruda'),(LAB+'_suav','suavizada')]):
            a=axs[3*n+1+k]; a.scatter(g.tiempo_seg,np.zeros(len(g)),c=g[col],cmap=cmap,vmin=0,vmax=4,marker='|',s=300,lw=1.2); a.set_yticks([]); a.set_ylabel(lab,rotation=0,ha='right',va='center',fontsize=6.5); a.grid(False)
            for s in a.spines.values(): s.set_visible(False)
    axs[-1].set_xlabel('tiempo (s)'); fig.legend([mp.Patch(color=CL[j]) for j in range(K)],NAMES,ncol=5,fontsize=6.5,loc='lower center',bbox_to_anchor=(.5,-.02)); fig.tight_layout(rect=(0,.03,1,1)); save(fig,'fig28_lineas_tiempo')
# ---- Fig 29 permanencia y transiciones
if want('29'):
    rr=pd.read_pickle('work/runs_raw.pkl'); rs=pd.read_pickle('work/runs_sm.pkl'); tr=np.load('work/trans.npy'); fig,axs=plt.subplots(1,2,figsize=(6.8,2.9)); bins=np.logspace(np.log10(1/120),np.log10(30),40)
    axs[0].hist(rr.dur,bins=bins,alpha=.65,color='#B6465F',label='etiquetas crudas'); axs[0].hist(rs.dur,bins=bins,alpha=.65,color='#2E6F9E',label='suavizadas (0,5 s)'); axs[0].set_xscale('log'); axs[0].set_xlabel('duración del episodio (s)'); axs[0].set_ylabel('episodios'); axs[0].legend(fontsize=7); axs[0].set_title('Permanencia en cada estado')
    P=tr/tr.sum(1,keepdims=True); im=axs[1].imshow(P,cmap='Blues',vmin=0,vmax=.6); axs[1].grid(False); axs[1].set_xticks(range(K)); axs[1].set_yticks(range(K)); axs[1].set_xticklabels([f'C{i+1}' for i in range(K)]); axs[1].set_yticklabels([f'C{i+1}' for i in range(K)]); axs[1].set_xlabel('estado siguiente'); axs[1].set_ylabel('estado de origen'); axs[1].set_title('Transiciones entre episodios')
    for i in range(K):
        for j in range(K):
            if i!=j: axs[1].text(j,i,f'{P[i,j]:.2f}',ha='center',va='center',fontsize=7,color='w' if P[i,j]>.35 else 'k')
    fig.tight_layout(); save(fig,'fig29_permanencia'); pd.DataFrame(P).to_pickle('work/trans_p.pkl')
# ---- Fig 30 NMI
if want('30'):
    nm=pd.read_pickle('work/nmi.pkl'); fig,ax=plt.subplots(figsize=(5.4,2.7)); w=.35
    for i,(c,col) in enumerate([('BASE','#B6465F'),('MEJ','#2E6F9E')]):
        g=nm[nm.conjunto==c]; ax.bar(np.arange(3)+(i-.5)*w,g.nmi_director,w,color=col,label={'BASE':'BASE: posiciones absolutas','MEJ':'MEJ: marco corporal'}[c])
        for x,v in zip(np.arange(3)+(i-.5)*w,g.nmi_director): ax.text(x,v+.01,f'{v:.2f}',ha='center',fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(['k = 2','k = 3','k = 4']); ax.set_ylabel('información mutua normalizada\ncluster–director'); ax.legend(fontsize=7); ax.set_title('¿Cuánto del cluster se explica por quién dirige?'); fig.tight_layout(); save(fig,'fig30_nmi')
# ---- Fig 31 batuta
if want('31'):
    B=D[D.batuta_velocidad.notna()]; s=B.sample(60000,random_state=1); fig,axs=plt.subplots(1,2,figsize=(6.8,2.8),gridspec_kw={'width_ratios':[1.1,1]})
    axs[0].hexbin(s.mano_der_velocidad,s.batuta_velocidad,gridsize=45,extent=(0,3,0,5),bins='log',cmap='viridis',mincnt=1); axs[0].plot([0,3],[0,3],'w--',lw=.8); axs[0].set_xlabel('velocidad mano derecha (m/s)'); axs[0].set_ylabel('velocidad extremo de batuta (m/s)'); axs[0].set_title('Batuta vs. mano derecha')
    dirs=sorted(B.director.unique()); bp=axs[1].boxplot([B[B.director==d].batuta_velocidad.clip(upper=6) for d in dirs],tick_labels=[f'D{d}' for d in dirs],patch_artist=True,showfliers=False,medianprops=dict(color='k'),widths=.55)
    for p,d in zip(bp['boxes'],dirs): p.set_facecolor(DIRC[d-1]); p.set_alpha(.6)
    axs[1].set_ylabel('velocidad de la batuta (m/s)'); axs[1].set_title('Por director (el D5 no usó batuta)'); fig.tight_layout(); save(fig,'fig31_batuta')
# ---- Fig 32 huella por toma
if want('32'):
    t=pd.crosstab(D.recording_id,D.cl,normalize='index'); order=tk.sort_values(['director','compositor','recording_id']).recording_id; t=t.loc[order]
    fig,ax=plt.subplots(figsize=(5.0,7.4)); im=ax.imshow(t.values,cmap='Blues',vmin=0,vmax=1,aspect='auto'); ax.grid(False); ax.set_xticks(range(K)); ax.set_xticklabels([n.split()[0] for n in NAMES])
    ax.set_yticks(range(len(t))); ax.set_yticklabels([r.replace('_trial','·t').replace('Conductor','D') for r in t.index],fontsize=5.2)
    dd=tk.set_index('recording_id').loc[t.index].director.values
    for i in range(1,len(dd)):
        if dd[i]!=dd[i-1]: ax.axhline(i-.5,color='k',lw=1)
    fig.colorbar(im,ax=ax,fraction=.04,pad=.02,label='proporción de la toma'); ax.set_title('Huella de cada toma (agrupadas por director)'); fig.tight_layout(); save(fig,'fig32_huella_tomas')
print('batch2 done')
