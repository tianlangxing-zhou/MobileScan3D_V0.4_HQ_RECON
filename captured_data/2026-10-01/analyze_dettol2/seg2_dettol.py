import sys, json, math
from collections import defaultdict, Counter

def load_ply(path):
    pts=[]
    with open(path,'rb') as f:
        # read header (ascii)
        line=f.readline()
        while line:
            s=line.decode('latin1').strip()
            if s.startswith('element vertex'):
                n=int(s.split()[2]); break
            line=f.readline()
        # skip to end_header
        for _ in range(100):
            l=f.readline().decode('latin1').strip()
            if l=='end_header': break
        for _ in range(n):
            l=f.readline().decode('latin1').strip()
            if not l: continue
            parts=l.split()
            if len(parts)<3: continue
            x,y,z=float(parts[0]),float(parts[1]),float(parts[2])
            pts.append((x,y,z))
    return pts

def main():
    base=sys.argv[1]
    ply=base+"_debug.ply"
    pts=load_ply(ply)
    print(f"cloud pts={len(pts)}")
    if not pts: return
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; zs=[p[2] for p in pts]
    print(f"raw bbox X={max(xs)-min(xs):.3f} Y={max(ys)-min(ys):.3f} Z={max(zs)-min(zs):.3f}")

    # table height = mode of y via histogram
    lo,hi=min(ys),max(ys)
    nb=200; bw=(hi-lo)/nb
    hist=Counter()
    for y in ys:
        b=int((y-lo)/bw); b=min(nb-1,max(0,b)); hist[b]+=1
    peak=max(hist, key=hist.get)
    tableY=lo+(peak+0.5)*bw
    print(f"tableY(mode)={tableY:.4f}  (peak bin count={hist[peak]})")

    # height above table
    H=[(p[1]-tableY) for p in pts]
    # grid max-height map (x,z), cell 0.01
    cell=0.01
    grid=defaultdict(float)   # (gx,gz)-> maxH
    idx_by_cell=defaultdict(list)
    for i,p in enumerate(pts):
        gx=int(round(p[0]/cell)); gz=int(round(p[2]/cell))
        h=H[i]
        if h>grid[(gx,gz)]: grid[(gx,gz)]=h
        idx_by_cell[(gx,gz)].append(i)
    # footprint = cells whose maxHeight > 0.12 (bottle reaches 0.283)
    thr=0.12
    footprint=set(c for c,h in grid.items() if h>thr)
    if not footprint:
        print("no footprint cells > thr, lower thr"); thr=0.06; footprint=set(c for c,h in grid.items() if h>thr)
    print(f"footprint cells(maxH>{thr:.2f})={len(footprint)}")
    # connected components (4-neigh) over footprint
    seen=set(); comps=[]
    from collections import deque
    for c in footprint:
        if c in seen: continue
        q=deque([c]); seen.add(c); cur=[c]
        while q:
            x,z=q.popleft()
            for dx,dz in ((1,0),(-1,0),(0,1),(0,-1)):
                n=(x+dx,z+dz)
                if n in footprint and n not in seen:
                    seen.add(n); q.append(n); cur.append(n)
        comps.append(cur)
    comps.sort(key=len, reverse=True)
    print(f"connected components={len(comps)}  sizes(top5)={ [len(c) for c in comps[:5]] }")
    # report each component's bbox (use ALL pts within footprint cells, i.e. includes base)
    real=[0.283,0.277,0.197]
    for ci,comp in enumerate(comps[:3]):
        cellset=set(comp)
        sel=[i for i in range(len(pts)) if (int(round(pts[i][0]/cell)),int(round(pts[i][2]/cell))) in cellset]
        if not sel: continue
        sx=[pts[i][0] for i in sel]; sy=[pts[i][1] for i in sel]; sz=[pts[i][2] for i in sel]
        span=(max(sx)-min(sx), max(sy)-min(sy), max(sz)-min(sz))
        n=len(sel)
        # sorted-match k
        order=sorted(range(3), key=lambda a:-span[a])
        ks=[real[r]/span[order[r]] for r in range(3)]
        print(f"  comp#{ci} n={n} spanX={span[0]:.3f} Y={span[1]:.3f} Z={span[2]:.3f}  sortedMatch k={ [round(k,3) for k in ks] } mean={sum(ks)/3:.3f}")

if __name__=='__main__':
    main()
