import numpy as np, pandas as pd, json, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mp
from matplotlib.colors import LinearSegmentedColormap
from mocap_pipeline import *
warnings.filterwarnings('ignore')
OUT='outputs/figuras/'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9.5,'axes.labelsize':8.5,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.25,'grid.linewidth':.5,'legend.frameon':False,'figure.dpi':100,'savefig.dpi':200,'savefig.bbox':'tight'})
CL=['#2E6F9E','#E08E2B','#4C9F70','#B6465F','#7C5BA6']; NAMES=['C1 Compacto','C2 Frontal moderado','C3 Amplio','C4 Asimétrico','C5 Intenso']
COMP={'Bartok':'#C0504D','Dvorak':'#4F81BD','Mozart':'#9BBB59'}
def save(fig,n): fig.savefig(OUT+n+'.png'); plt.close(fig); print('ok',n,flush=True)
cols=json.load(open('work/cols.json')); ci={c:i for i,c in enumerate(cols)}
M=np.load('work/raw.npy',mmap_mode='r'); rid=np.load('work/rid.npy')
def raw(r):
    m=np.flatnonzero(rid==r); X=np.asarray(M[m[0]:m[-1]+1]); return X[np.argsort(X[:,ci['frame']],kind='stable')]
sel=json.load(open('work/seleccion.json')); fusion=json.load(open('work/fusion.json')); F=pd.read_pickle('work/features_sel.pkl'); D=pd.read_pickle('work/D_labels.pkl')
FA=pd.read_pickle('work/features_all.pkl')
# ---------- Fig 1: pipeline
fig,ax=plt.subplots(figsize=(6.8,3.1)); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,46)
def box(x,y,t,c): 
    ax.add_patch(mp.FancyBboxPatch((x,y),21,11,boxstyle='round,pad=0.3',fc=c,ec='#444',lw=.8)); ax.text(x+10.5,y+5.5,t,ha='center',va='center',fontsize=7)
r1=[('62 archivos C3D\n120 Hz, 20 marcadores',2,'#DDE8F3'),('Extracción y fusión\nde variantes _1/_2',27,'#DDE8F3'),('Limpieza: oclusiones\ny suavizado SG',52,'#FBE5C8'),('Variables\nbiomecánicas (26)',77,'#FBE5C8')]
r2=[('Clustering\nK-Means · Ward · GMM',2,'#D8EBDD'),('Evaluación: Silhouette,\nDavies-Bouldin, CH, ARI',27,'#E9DDF0'),('Interpretación: perfiles,\ndirector, tiempo',52,'#E9DDF0')]
for t,x,c in r1: box(x,30,t,c)
for t,x,c in r2: box(x,8,t,c)
for x in [23.3,48.3,73.3]: ax.annotate('',xy=(x+3.2,35.5),xytext=(x,35.5),arrowprops=dict(arrowstyle='->',lw=1))
for x in [23.3,48.3]: ax.annotate('',xy=(x+3.2,13.5),xytext=(x,13.5),arrowprops=dict(arrowstyle='->',lw=1))
ax.plot([87.5,87.5],[30,24],color='k',lw=1); ax.plot([87.5,12.5],[24,24],color='k',lw=1); ax.annotate('',xy=(12.5,19.5),xytext=(12.5,24),arrowprops=dict(arrowstyle='->',lw=1))
ax.text(50,43,'Fases CRISP-DM: datos → preparación → modelado → evaluación → despliegue',ha='center',fontsize=7.5,style='italic')
ax.grid(False); save(fig,'fig01_pipeline')
# ---------- Fig 2: marcadores en marco corporal (toma de ejemplo)
r='Mozart_Conductor02_trial01'; X=raw(r); P={}
for m in MARKERS:
    P[m],_=clean_marker(X[:,[ci[f'{m}_x'],ci[f'{m}_y'],ci[f'{m}_z']]])
T=np.stack([P[m] for m in ['C7','CLAV','STRN','T10']]); torso=np.nanmean(T,0)
sh=P['RSHO']-P['LSHO']; lat=np.stack([sh[:,0],sh[:,1],np.zeros(len(sh))],1); lat/=np.linalg.norm(lat,axis=1,keepdims=True); fwd=np.stack([-lat[:,1],lat[:,0],np.zeros(len(sh))],1)
Q={}
for m in MARKERS:
    rel=P[m]-torso; Q[m]=np.array([np.nanmean(np.sum(rel*lat,1)),np.nanmean(np.sum(rel*fwd,1)),np.nanmean(rel[:,2])])
fig,axs=plt.subplots(1,2,figsize=(6.6,3.6))
segs=[('LSHO','RSHO'),('LSHO','LELB'),('LELB','LWRA'),('RSHO','RELB'),('RELB','RWRA'),('LWRA','LFIN'),('RWRA','RFIN'),('C7','T10'),('CLAV','STRN'),('LFHD','RFHD'),('LBHD','RBHD'),('RWRA','Baton Shaft'),('Baton Shaft','Baton End')]
grp={'Cabeza':['LFHD','RFHD','LBHD','RBHD'],'Hombros':['LSHO','RSHO'],'Codos':['LELB','RELB'],'Muñecas':['LWRA','LWRB','RWRA','RWRB'],'Dedos':['LFIN','RFIN'],'Batuta':['Baton End','Baton Shaft'],'Torso':['C7','CLAV','STRN','T10']}
gc={'Cabeza':'#7f7f7f','Hombros':'#2E6F9E','Codos':'#E08E2B','Muñecas':'#B6465F','Dedos':'#7C5BA6','Batuta':'#4C9F70','Torso':'#444444'}
for ax,(ia,ib,lab) in zip(axs,[(0,2,'Vista frontal (lateral, vertical)'),(1,2,'Vista lateral (frontal, vertical)')]):
    for a,b in segs:
        ax.plot([Q[a][ia],Q[b][ia]],[Q[a][ib],Q[b][ib]],color='#bbb',lw=1,zorder=1)
    for g,ms in grp.items():
        ax.scatter([Q[m][ia] for m in ms],[Q[m][ib] for m in ms],s=22,color=gc[g],label=g,zorder=3)
    ax.set_title(lab); ax.set_xlabel('lateral (m)' if ia==0 else 'frontal (m)'); ax.set_ylabel('vertical respecto al torso (m)'); ax.set_aspect('equal')
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=7,fontsize=7,bbox_to_anchor=(0.5,-0.04)); save(fig,'fig02_marcadores')
# ---------- Fig 3: ilustración didáctica de algoritmos
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
Xb,_=make_blobs(150,centers=[(0,0),(4,1),(1.5,4)],cluster_std=[.9,.8,.9],random_state=3)
fig,axs=plt.subplots(1,3,figsize=(6.8,2.3))
axs[0].scatter(Xb[:,0],Xb[:,1],s=8,color='#777'); axs[0].set_title('Datos sin etiquetas')
km=KMeans(3,n_init=10,random_state=0).fit(Xb); axs[1].scatter(Xb[:,0],Xb[:,1],c=[CL[i] for i in km.labels_],s=8); axs[1].scatter(*km.cluster_centers_.T,marker='X',s=70,c='k'); axs[1].set_title('K-Means (centroides)')
Zl=linkage(Xb,'ward'); dendrogram(Zl,ax=axs[2],no_labels=True,color_threshold=Zl[-3,2],above_threshold_color='#999'); axs[2].set_title('Dendrograma de Ward'); axs[2].set_ylabel('distancia de fusión')
for a in axs[:2]: a.set_xticks([]); a.set_yticks([])
save(fig,'fig03_algoritmos')
# ---------- Fig 4: composición del conjunto
tk=pd.read_pickle('work/tomas.pkl'); allr=FA.groupby(['recording_id','compositor','director']).size().reset_index(name='n')
fig,axs=plt.subplots(1,2,figsize=(6.8,2.8))
ct=pd.crosstab(tk.director,tk.compositor); im=axs[0].imshow(ct.values,cmap='Blues',aspect='auto'); axs[0].set_xticks(range(3)); axs[0].set_xticklabels(['Bartók','Dvořák','Mozart']); axs[0].set_yticks(range(6)); axs[0].set_yticklabels([f'Director {i}' for i in ct.index]); axs[0].grid(False)
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]): axs[0].text(j,i,ct.values[i,j],ha='center',va='center',fontsize=8)
axs[0].set_title('Tomas únicas por director y obra')
dur=tk.groupby('compositor').n.sum()/FPS/60; axs[1].bar(['Bartók','Dvořák','Mozart'],[dur['Bartok'],dur['Dvorak'],dur['Mozart']],color=[COMP['Bartok'],COMP['Dvorak'],COMP['Mozart']]); axs[1].set_ylabel('minutos de fotogramas válidos'); axs[1].set_title('Duración válida por obra')
save(fig,'fig04_composicion')
# ---------- Fig 5: mapa de oclusión
occ=pd.read_pickle('work/occ.pkl').drop(columns=['n','nan']); disc=set(r for v in fusion.values() for r in v['variantes'])
fig,ax=plt.subplots(figsize=(6.6,8.6)); im=ax.imshow(occ.values,cmap='magma_r',vmin=0,vmax=1,aspect='auto'); ax.grid(False)
ax.set_xticks(range(20)); ax.set_xticklabels(occ.columns,rotation=90,fontsize=7); ax.set_yticks(range(len(occ))); ax.set_yticklabels([('‡ ' if r in disc else '')+r.replace('_',' ') for r in occ.index],fontsize=5.6)
for t,r in zip(ax.get_yticklabels(),occ.index):
    if r in disc: t.set_color('#B22')
cb=fig.colorbar(im,ax=ax,fraction=.03,pad=.02); cb.set_label('fracción de fotogramas ocluidos'); ax.set_title('Oclusión por toma y marcador (‡ = archivo de un par _1/_2 de la misma toma)')
save(fig,'fig05_oclusion')
# ---------- Fig 6: huecos
ci_=json.load(open('work/clean_info.json')); g=[x for r in sel['seleccionadas'] for m in MARKERS for x in ci_[r][m]['longitudes']]
g=np.array(g); fig,axs=plt.subplots(1,2,figsize=(6.8,2.7))
bins=np.unique(np.round(np.logspace(0,4.4,30)).astype(int)); axs[0].hist(g,bins=bins,color='#2E6F9E'); axs[0].set_xscale('log'); axs[0].set_yscale('log'); axs[0].axvline(12,color='#B22',ls='--'); axs[0].text(13,axs[0].get_ylim()[1]*.5,'límite = 12',color='#B22',fontsize=7.5); axs[0].set_xlabel('longitud del hueco (frames)'); axs[0].set_ylabel('número de huecos'); axs[0].set_title('Distribución de longitudes de hueco')
fr=np.array([g[g<=12].sum(),g[g>12].sum()]); axs[1].bar(['≤ 12 frames\n(interpolables)','> 12 frames\n(se conservan como NaN)'],fr,color=['#4C9F70','#B6465F']); axs[1].set_ylabel('fotogramas-marcador ocluidos'); axs[1].set_title('¿Dónde está la información perdida?')
for i,v in enumerate(fr): axs[1].text(i,v,f'{v/fr.sum()*100:.1f} %',ha='center',va='bottom',fontsize=8)
save(fig,'fig06_huecos')
# ---------- Fig 7: evidencia pares _1/_2
a,b='Dvorak_Conductor03_trial01_1','Dvorak_Conductor03_trial01_2'; A,B=raw(a),raw(b)
fig,axs=plt.subplots(8,1,figsize=(6.8,3.3),sharex=True); k=0; tt=np.arange(len(A))/FPS
for mk in ['LELB','RELB','LWRA','RWRA']:
    for nm,X in [('_1',A),('_2',B)]:
        c=[ci[f'{mk}_x'],ci[f'{mk}_y'],ci[f'{mk}_z']]; ok=np.abs(X[:,c]).sum(1)>0
        axs[k].fill_between(tt,0,1,where=ok,color='#2E6F9E' if nm=='_1' else '#E08E2B',step='mid',lw=0); axs[k].set_yticks([]); axs[k].set_ylabel(f'{mk}{nm}',rotation=0,ha='right',va='center',fontsize=6.5); axs[k].grid(False)
        for s in axs[k].spines.values(): s.set_visible(False)
        k+=1
axs[-1].set_xlabel('tiempo (s)'); axs[0].set_title('Presencia de marcadores en las dos variantes de una misma toma (azul: _1, naranja: _2)',fontsize=8.5)
save(fig,'fig07_pares')
