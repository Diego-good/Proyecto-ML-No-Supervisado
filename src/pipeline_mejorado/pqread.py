import struct, numpy as np
from thrift_pq import R, footer

def snappy_decompress(b):
    r=R(b); n=r.varint(); out=bytearray(n); o=0; p=r.p; L=len(b)
    while p<L:
        t=b[p]; p+=1; tag=t&3
        if tag==0:
            ln=t>>2
            if ln>=60:
                nb=ln-59; ln=int.from_bytes(b[p:p+nb],'little'); p+=nb
            ln+=1
            out[o:o+ln]=b[p:p+ln]; p+=ln; o+=ln
        else:
            if tag==1: ln=((t>>2)&7)+4; off=((t>>5)<<8)|b[p]; p+=1
            elif tag==2: ln=(t>>2)+1; off=int.from_bytes(b[p:p+2],'little'); p+=2
            else: ln=(t>>2)+1; off=int.from_bytes(b[p:p+4],'little'); p+=4
            s=o-off
            if off>=ln: out[o:o+ln]=out[s:s+ln]
            else:
                pat=bytes(out[s:o]); reps=(ln//off)+1
                out[o:o+ln]=(pat*reps)[:ln]
            o+=ln
    assert o==n,(o,n)
    return bytes(out)

def hybrid(buf,p,bw,count):
    out=np.empty(count,dtype=np.int64); n=0
    mask=(1<<bw)-1
    while n<count:
        h=0;sh=0
        while True:
            x=buf[p];p+=1;h|=(x&0x7f)<<sh
            if not x&0x80:break
            sh+=7
        if h&1:
            groups=h>>1; nv=groups*8; nbytes=groups*bw
            bits=np.unpackbits(np.frombuffer(buf,dtype=np.uint8,count=nbytes,offset=p),bitorder='little')
            vals=bits[:nv*bw].reshape(nv,bw)@(1<<np.arange(bw)) if bw>0 else np.zeros(nv,int)
            p+=nbytes; take=min(nv,count-n); out[n:n+take]=vals[:take]; n+=take
        else:
            run=h>>1; bn=(bw+7)//8; v=int.from_bytes(buf[p:p+bn],'little')&mask if bn else 0; p+=bn
            take=min(run,count-n); out[n:n+take]=v; n+=take
    return out,p

DT={1:np.dtype('<i4'),2:np.dtype('<i8'),4:np.dtype('<f4'),5:np.dtype('<f8')}
def read_column(path,ft,rg,ci):
    meta=rg[1][ci][3]; typ=meta[1]; nvals=meta[5]; name=meta[3][0].decode()
    start=meta.get(11) or meta[9]
    if 11 in meta and meta[11]<meta[9]: start=meta[11]
    csize=meta[7]
    f=open(path,'rb'); f.seek(start); raw=f.read(csize)
    p=0; dic=None; chunks=[]
    total=0
    while total<nvals and p<len(raw):
        rr=R(raw,p); ph=rr.struct(); p=rr.p
        ptype=ph[1]; usz=ph[2]; csz=ph[3]
        page=snappy_decompress(raw[p:p+csz]) if meta[4]==1 else raw[p:p+csz]; p+=csz
        if ptype==2:
            nd=ph[7][1]
            if typ==6:
                d=[];q=0
                for _ in range(nd):
                    l=struct.unpack_from('<I',page,q)[0]; q+=4; d.append(page[q:q+l].decode()); q+=l
                dic=np.array(d,dtype=object)
            else: dic=np.frombuffer(page,dtype=DT[typ],count=nd)
        elif ptype==0:
            dh=ph[5]; nv=dh[1]; enc=dh[2]; q=0
            # definition levels (optional col, max level 1)
            l=struct.unpack_from('<I',page,q)[0]; q+=4
            dl,_=hybrid(page,q,1,nv); q+=l
            nn=int((dl==1).sum())
            if enc in(2,8):
                bw=page[q]; q+=1
                idx,_=hybrid(page,q,bw,nn); vals=dic[idx]
            elif enc==0:
                vals=np.frombuffer(page,dtype=DT[typ],count=nn,offset=q)
            else: raise ValueError(enc)
            if nn<nv:
                full=np.full(nv,np.nan) if typ in(4,5) else np.full(nv,None,dtype=object); full[dl==1]=vals; vals=full
            chunks.append(vals); total+=nv
        else: raise ValueError(ptype)
    return name,np.concatenate(chunks)

def read_parquet(path):
    ft=footer(path); rg=ft[4][0]; cols={}
    for ci in range(len(rg[1])):
        n,a=read_column(path,ft,rg,ci); cols[n]=a
    return cols
if __name__=='__main__':
    import time,sys; t=time.time()
    ft=footer('data/datos_crudos.parquet'); rg=ft[4][0]
    n,a=read_column('data/datos_crudos.parquet',ft,rg,0); print(n,a[:3],len(a),len(set(a)),time.time()-t)
    n,a=read_column('data/datos_crudos.parquet',ft,rg,3); print(n,a[:5],len(a),np.nanmin(a),np.nanmax(a),time.time()-t)
