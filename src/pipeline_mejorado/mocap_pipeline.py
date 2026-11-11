#!/usr/bin/env python3
"""Etapa 1 del pipeline mejorado: limpieza y extracción de variables biomecánicas.

Cambios respecto al script original (features_clustering.py):
  1. Solo se interpolan huecos internos de hasta MAX_GAP frames. Los huecos más largos
     permanecen como NaN (el original los rellenaba con ffill/bfill, congelando la posición).
  2. El suavizado Savitzky-Golay se aplica por tramos contiguos válidos (sin cruzar huecos).
  3. Muñeca = punto medio de WRA y WRB; posiciones expresadas en un marco corporal
     (lateral, frontal, vertical) referido al torso, para eliminar la ubicación y la orientación
     del director en el espacio de captura.
  4. Solo se agrupan fotogramas completos (todas las variables válidas); no se mezclan
     grabaciones al imputar.
"""
import json, re
import numpy as np, pandas as pd
from scipy.signal import savgol_filter

FPS = 120.0
MAX_GAP = 12          # frames (0,1 s)
SG_WIN, SG_POLY = 9, 2  # 0,075 s
MARKERS = ['LFHD','RFHD','LBHD','RBHD','LSHO','RSHO','LELB','RELB','LWRA','LWRB','RWRA','RWRB',
           'LFIN','RFIN','Baton End','Baton Shaft','C7','CLAV','STRN','T10']

def nan_runs(isn):
    d = np.diff(np.concatenate([[0], isn.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)   # [start, end)

def clean_marker(xyz, max_gap=MAX_GAP, win=SG_WIN, poly=SG_POLY, fps=FPS):
    """xyz: (N,3) en metros. Devuelve (limpio, info)."""
    a = xyz.astype(float).copy()
    zero = (np.abs(a).sum(1) == 0) | np.isnan(a).any(1)
    a[zero] = np.nan
    info = {'ocluidos': int(zero.sum()), 'huecos': 0, 'huecos_interpolados': 0, 'frames_interpolados': 0,
            'max_hueco': 0, 'longitudes': []}
    isn = np.isnan(a[:, 0])
    s, e = nan_runs(isn)
    info['huecos'] = len(s)
    for i, j in zip(s, e):
        L = j - i; info['longitudes'].append(int(L)); info['max_hueco'] = max(info['max_hueco'], int(L))
        if L <= max_gap and i > 0 and j < len(a):
            for c in range(3):
                a[i:j, c] = np.interp(np.arange(i, j), [i - 1, j], [a[i - 1, c], a[j, c]])
            info['huecos_interpolados'] += 1; info['frames_interpolados'] += int(L)
    # suavizado por tramos válidos
    valid = ~np.isnan(a[:, 0])
    ns, ne = nan_runs(~valid)  # runs of NaN; invertimos para obtener tramos válidos
    vs, ve = nan_runs(valid)
    out = np.full_like(a, np.nan)
    for i, j in zip(vs, ve):
        if j - i >= win:
            out[i:j] = savgol_filter(a[i:j], win, poly, axis=0, mode='interp')
    info['residual_nan'] = int(np.isnan(out[:, 0]).sum())
    return out, info

def vel_acc(pos, dt):
    v = np.gradient(pos, dt, axis=0); a = np.gradient(v, dt, axis=0)
    return v, a

def angle3(a, b, c):
    u, v = a - b, c - b
    den = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        cos = np.sum(u * v, axis=1) / den
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def process_recording(X, colidx, fps=FPS, max_gap=MAX_GAP, win=SG_WIN):
    """X: matriz (N, ncols) de una grabación. Devuelve (DataFrame de variables, info por marcador)."""
    dt = 1.0 / fps; N = len(X); info = {}; P = {}
    for m in MARKERS:
        c = [colidx[f'{m}_x'], colidx[f'{m}_y'], colidx[f'{m}_z']]
        P[m], info[m] = clean_marker(X[:, c], max_gap=max_gap, win=win, fps=fps)
    wl = (P['LWRA'] + P['LWRB']) / 2; wr = (P['RWRA'] + P['RWRB']) / 2   # NaN si falta alguno
    pts = {'mano_izq': wl, 'mano_der': wr, 'codo_izq': P['LELB'], 'codo_der': P['RELB']}
    # torso de referencia (mínimo 2 marcadores)
    T = np.stack([P[m] for m in ['C7', 'CLAV', 'STRN', 'T10']])
    cnt = np.sum(~np.isnan(T[:, :, 0]), axis=0)
    with np.errstate(all='ignore'):
        torso = np.nanmean(T, axis=0)
    torso[cnt < 2] = np.nan
    # marco corporal: lateral = hombro izq -> der proyectado en XY; frontal = z x lateral; vertical = z
    sh = P['RSHO'] - P['LSHO']; lat = np.stack([sh[:, 0], sh[:, 1], np.zeros(N)], 1)
    nl = np.linalg.norm(lat, axis=1, keepdims=True)
    with np.errstate(all='ignore'):
        lat = lat / nl
    fwd = np.stack([-lat[:, 1], lat[:, 0], np.zeros(N)], 1)   # z x lat
    f = {}
    for key, p in pts.items():
        v, a = vel_acc(p, dt)
        f[f'{key}_velocidad'] = np.linalg.norm(v, axis=1)
        f[f'{key}_aceleracion'] = np.linalg.norm(a, axis=1)
        f[f'{key}_energia_rel'] = 0.5 * np.sum(v ** 2, axis=1)
        for i, ax in enumerate('xyz'):
            f[f'{key}_abs_{ax}'] = p[:, i]
        rel = p - torso
        f[f'{key}_lat'] = np.sum(rel * lat, axis=1)
        f[f'{key}_fre'] = np.sum(rel * fwd, axis=1)
        f[f'{key}_ver'] = rel[:, 2]
    f['mano_izq_dist_torso'] = np.linalg.norm(wl - torso, axis=1)
    f['mano_der_dist_torso'] = np.linalg.norm(wr - torso, axis=1)
    f['angulo_codo_izq'] = angle3(P['LSHO'], P['LELB'], wl)
    f['angulo_codo_der'] = angle3(P['RSHO'], P['RELB'], wr)
    f['velocidad_media_manos'] = (f['mano_izq_velocidad'] + f['mano_der_velocidad']) / 2
    f['asimetria_velocidad_manos'] = np.abs(f['mano_izq_velocidad'] - f['mano_der_velocidad'])
    # batuta (solo descriptivo)
    be, bs = P['Baton End'], P['Baton Shaft']
    if not np.isnan(be[:, 0]).all():
        v, a = vel_acc(be, dt); f['batuta_velocidad'] = np.linalg.norm(v, axis=1)
        d = be - bs; f['batuta_longitud'] = np.linalg.norm(d, axis=1)
    else:
        f['batuta_velocidad'] = np.full(N, np.nan); f['batuta_longitud'] = np.full(N, np.nan)
    df = pd.DataFrame(f).replace([np.inf, -np.inf], np.nan)
    return df, info

FEATS_BASE = (['%s_%s' % (k, s) for k in ['mano_izq', 'mano_der', 'codo_izq', 'codo_der'] for s in ['velocidad', 'aceleracion', 'energia_rel', 'abs_x', 'abs_y', 'abs_z']]
              + ['mano_izq_dist_torso', 'mano_der_dist_torso', 'angulo_codo_izq', 'angulo_codo_der', 'velocidad_media_manos', 'asimetria_velocidad_manos'])
FEATS_MEJ = (['%s_%s' % (k, s) for k in ['mano_izq', 'mano_der', 'codo_izq', 'codo_der'] for s in ['velocidad', 'aceleracion', 'lat', 'fre', 'ver']]
             + ['mano_izq_dist_torso', 'mano_der_dist_torso', 'angulo_codo_izq', 'angulo_codo_der', 'velocidad_media_manos', 'asimetria_velocidad_manos'])

def parse_id(r):
    m = re.match(r'(Bartok|Dvorak|Mozart)_Conductor(\d+)_trial(\d+)(?:_(\d+))?', r)
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4) or 0)

def merge_variants(XA, XB, colidx):
    """Fusiona dos variantes de la misma toma: por marcador, usa la variante con menos oclusión
    y completa sus fotogramas ocluidos (ceros XYZ o NaN) con la otra variante."""
    X = XA.copy(); stats = {}
    for m in MARKERS:
        c = [colidx[f'{m}_x'], colidx[f'{m}_y'], colidx[f'{m}_z']]
        a, b = XA[:, c], XB[:, c]
        za = (np.abs(np.nan_to_num(a)).sum(1) == 0) | np.isnan(a).any(1)
        zb = (np.abs(np.nan_to_num(b)).sum(1) == 0) | np.isnan(b).any(1)
        if zb.sum() < za.sum():
            a, b, za, zb = b, a, zb, za
        out = a.copy(); fill = za & ~zb; out[fill] = b[fill]
        X[:, c] = out; stats[m] = int(fill.sum())
    return X, stats

if __name__ == '__main__':
    import time
    t = time.time()
    cols = json.load(open('work/cols.json')); colidx = {c: i for i, c in enumerate(cols)}
    M = np.load('work/raw.npy', mmap_mode='r'); rid = np.load('work/rid.npy')
    u, idx = np.unique(rid, return_index=True); u = u[np.argsort(idx)]
    def getX(r):
        m = np.flatnonzero(rid == r); X = np.asarray(M[m[0]:m[-1] + 1])
        return X[np.argsort(X[:, colidx['frame']], kind='stable')]
    bases = {}
    for r in u: bases.setdefault(re.sub(r'_[12]$', '', r), []).append(r)
    frames = []; infos = {}; fusion = {}
    for b, rs in bases.items():
        if len(rs) == 2:
            XA, XB = getX(rs[0]), getX(rs[1]); assert len(XA) == len(XB)
            X, st = merge_variants(XA, XB, colidx); fusion[b] = {'variantes': rs, 'frames_completados': st}
        else:
            X = getX(rs[0])
        df, info = process_recording(X, colidx)
        comp, cond, trial, take = parse_id(rs[0])
        df.insert(0, 'recording_id', b); df.insert(1, 'compositor', comp); df.insert(2, 'director', cond)
        df.insert(3, 'toma', trial * 10 + take); df.insert(4, 'frame', X[:, colidx['frame']].astype(int)); df.insert(5, 'tiempo_seg', X[:, colidx['tiempo_seg']])
        df['valido'] = df[FEATS_BASE + FEATS_MEJ].notna().all(axis=1)
        frames.append(df); infos[b] = info
        print(b, len(rs), len(df), round(df.valido.mean(), 3), round(time.time() - t, 1), flush=True)
    F = pd.concat(frames, ignore_index=True)
    num = F.select_dtypes('float64').columns
    F[num] = F[num].astype('float32')
    F.to_pickle('work/features_all.pkl')
    json.dump(infos, open('work/clean_info.json', 'w')); json.dump(fusion, open('work/fusion.json', 'w'))
    print('listo', F.shape, F.valido.mean(), F.valido.sum(), F.recording_id.nunique())
