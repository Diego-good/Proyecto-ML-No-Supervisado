import numpy as np, time, json
from pqread import read_parquet
t=time.time()
cols=read_parquet('data/datos_crudos.parquet')
print('loaded',len(cols),time.time()-t)
names=list(cols)
rid=cols['recording_id'].astype(str)
np.save('work/rid.npy',rid.astype('U40'))
num=[c for c in names if c!='recording_id']
M=np.column_stack([cols[c].astype(np.float64) for c in num])
np.save('work/raw.npy',M)
json.dump(num,open('work/cols.json','w'))
print(M.shape, M.nbytes/1e6,'MB', time.time()-t)
