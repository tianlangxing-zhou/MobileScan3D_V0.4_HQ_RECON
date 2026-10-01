import json, struct, sys, math
from collections import defaultdict, deque, Counter

def extract_positions(path):
    with open(path,'rb') as f:
        magic,ver,length=struct.unpack('<III',f.read(12))
        assert magic==0x46546C67,'not glb'
        clen,ctype=struct.unpack('<II',f.read(8))
        js=f.read(clen).decode('utf-8')
        # BIN chunk
        blen,btype=struct.unpack('<II',f.read(8))
        bin_data=f.read(blen)
    g=json.loads(js)
    acc=g['accessors']; meshes=g['meshes']
    # collect all POSITION accessors across primitives
    pos_lists=[]
    for mesh in meshes:
        for prim in mesh['primitives']:
            ai=prim['attributes']['POSITION']
            a=acc[ai]
            bv=g['bufferViews'][a['bufferView']]
            off=(bv.get('byteOffset',0)+a.get('byteOffset',0))
            cnt=a['count']
            fmt='<'+str(cnt*3)+'f'
            vals=struct.unpack_from(fmt, bin_data, off)
            pos_lists.append(vals)
    # flatten
    out=[]
    for vals in pos_lists:
        for i in range(0,len(vals),3):
            out.append((vals[i],vals[i+1],vals[i+2]))
    return out

def main():
    base=sys.argv[1]
    glb=base+'.glb'
    pts=extract_positions(glb)
    print(f"GLB verts={len(pts)}")
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; zs=[p[2] for p in pts]
    print(f"full bbox X={max(xs)-min(xs):.3f} Y={max(ys)-min(ys):.3f} Z={max(zs)-min(zs):.3f}")
    # table height mode
    lo,hi=min(ys),max(ys); nb=300; bw=(hi-lo)/nb
    hist=Counter()
    for y in ys:
        b=int((y-lo)/bw); b=min(nb-1,max(0,b)); hist[b]+=1
    peak=max(hist,key=hist.get); tableY=lo+(peak+0.5)*bw
    print(f"tableY(mode)={tableY:.4f} peakCount={hist[peak]} / {len(pts)}")
    # height map (x,z)
    cell=0.01
    grid=defaultdict(float); idxcell=defaultdict(list)
    for i,p in enumerate(pts):
        gx=int(round(p[0]/cell)); gz=int(round(p[2]/cell))
        h=p[1]-tableY
        if h>grid[(gx,gz)]: grid[(gx,gz)]=h
        idxcell[(gx,gz)].append(i)
    for thr in (0.12,0.08,0.05,0.03):
        fp=set(c for c,hh in grid.items() if hh>thr)
        if fp:
            break
    print(f"footprint cells(maxH>{thr:.2f})={len(fp)}")
    seen=set(); comps=[]
    for c in fp:
        if c in seen: continue
        q=deque([c]); seen.add(c); cur=[c]
        while q:
            x,z=q.popleft()
            for dx,dz in ((1,0),(-1,0),(0,1),(0,-1)):
                n=(x+dx,z+dz)
                if n in fp and n not in seen:
                    seen.add(n); q.append(n); cur.append(n)
        comps.append(cur)
    comps.sort(key=len,reverse=True)
    print(f"comps={len(comps)} sizes(top5)={[len(c) for c in comps[:5]]}")
    real=[0.283,0.277,0.197]
    for ci,comp in enumerate(comps[:3]):
        cs=set(comp)
        sel=[i for i in range(len(pts)) if (int(round(pts[i][0]/cell)),int(round(pts[i][2]/cell))) in cs]
        sx=[pts[i][0] for i in sel]; sy=[pts[i][1] for i in sel]; sz=[pts[i][2] for i in sel]
        span=(max(sx)-min(sx),max(sy)-min(sy),max(sz)-min(sz))
        order=sorted(range(3),key=lambda a:-span[a])
        ks=[real[r]/span[order[r]] for r in range(3)]
        print(f"  comp#{ci} n={len(sel)} spanX={span[0]:.3f} Y={span[1]:.3f} Z={span[2]:.3f} k(sorted)={[round(k,3) for k in ks]} mean={sum(ks)/3:.3f}")

if __name__=='__main__':
    main()
