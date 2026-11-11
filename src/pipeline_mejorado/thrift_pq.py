import struct
class R:
    def __init__(s,b,p=0): s.b=b; s.p=p
    def byte(s): v=s.b[s.p]; s.p+=1; return v
    def varint(s):
        r=0;sh=0
        while True:
            x=s.byte(); r|=(x&0x7f)<<sh
            if not x&0x80: return r
            sh+=7
    def zz(s): n=s.varint(); return (n>>1)^-(n&1)
    def binary(s): n=s.varint(); v=s.b[s.p:s.p+n]; s.p+=n; return v
    def value(s,t):
        if t in(1,2): return t==1
        if t==3: return struct.unpack('b',bytes([s.byte()]))[0]
        if t in(4,5,6): return s.zz()
        if t==7: v=struct.unpack('<d',s.b[s.p:s.p+8])[0]; s.p+=8; return v
        if t==8: return s.binary()
        if t in(9,10):
            h=s.byte(); n=h>>4; et=h&15
            if n==15: n=s.varint()
            return [s.value(et) for _ in range(n)]
        if t==12: return s.struct()
        raise ValueError(t)
    def struct(s):
        d={}; last=0
        while True:
            h=s.byte()
            if h==0: return d
            t=h&15; delta=h>>4
            fid=last+delta if delta else s.zz()
            last=fid
            d[fid]=s.value(t)
def footer(path):
    f=open(path,'rb'); f.seek(-8,2); n=struct.unpack('<I',f.read(4))[0]
    f.seek(-8-n,2); return R(f.read(n)).struct()
if __name__=='__main__':
    ft=footer('data/datos_crudos.parquet')
    print('rows',ft[3],'rowgroups',len(ft[4]))
    schema=ft[2]
    print([ (x.get(4,b'').decode(),x.get(1),x.get(3)) for x in schema[:6]])
    rg=ft[4][0]; print('rg rows',rg[3])
    cc=rg[1][0][3]; print({k:v for k,v in cc.items() if k!=12})
    print('codecs',{c[3][4] for r in ft[4] for c in r[1]}, 'encodings',{tuple(c[3][2]) for r in ft[4] for c in r[1]})
