import json, struct, sys, random, math
from collections import deque, defaultdict

random.seed(7)

def extract_positions_strided(path, stride=32):
    with open(path,'rb') as f:
        magic,ver,length=struct.unpack('<III',f.read(12))
        clen,ctype=struct.unpack('<II',f.read(8))
        js=f.read(clen).decode('utf-8')
        blen,btype=struct.unpack('<II',f.read(8))
        bin_data=f.read(blen)
    g=json.loads(js)
    a=g['accessors'][0]; bv=g['bufferViews'][a['bufferView']]
    off=bv.get('byteOffset',0)+a.get('byteOffset',0); cnt=a['count']
    pos=[]
    for i in range(cnt):
        b=off+i*stride
        x,y,z=struct.unpack_from('<fff',bin_data,b)
        pos.append((x,y,z))
    return pos

def ransac_plane(pts, iters=600, eps=0.012):
    n=len(pts); best=None; best_in=0
    for _ in range(iters):
        i,j,k=random.sample(range(n),3)
        p1,p2,p3=pts[i],pts[j],pts[k]
        ax,ay,az=p2[0]-p1[0],p2[1]-p1[1],p2[2]-p1[2]
        bx,by,bz=p3[0]-p1[0],p3[1]-p1[1],p3[2]-p1[2]
        nx=ay*bz-az*by; ny=az*bx-ax*bz; nz=ax*by-ay*bx
        ln=math.sqrt(nx*nx+ny*ny+nz*nz)
        if ln<1e-9: continue
        nx,ny,nz=nx/ln,ny/ln,nz/ln; d=-(nx*p1[0]+ny*p1[1]+nz*p1[2])
        inn=sum(1 for p in pts if abs(nx*p[0]+ny*p[1]+nz*p[2]+d)<eps)
        if inn>best_in: best_in=inn; best=(nx,ny,nz,d)
    return best,best_in

def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def norm(v):
    l=math.sqrt(dot(v,v)) or 1; return (v[0]/l,v[1]/l,v[2]/l)

def main():
    base=sys.argv[1]; glb=base+'.glb'
    pos=extract_positions_strided(glb,32)
    step=max(1,len(pos)//40000); sub=pos[::step]
    plane,inl=ransac_plane(sub, iters=600, eps=0.012)
    nx,ny,nz,d=plane
    print(f"plane normal=({nx:.3f},{ny:.3f},{nz:.3f}) d={d:.3f} inliers={100*inl/len(sub):.1f}%")
    # basis: u,v in plane
    up=(0,1,0)
    u=norm((ny*up[2]-nz*up[1], nz*up[0]-nx*up[2], nx*up[1]-ny*up[0]))
    if dot(u,u)<0.1: u=norm((1,0,0))
    v=norm((ny*u[2]-nz*u[1], nz*u[0]-nx*u[2], nx*u[1]-ny*u[0]))
    # signed height
    h=[nx*p[0]+ny*p[1]+nz*p[2]+d for p in pos]
    # bottle = points with height above table (0.01..0.45)
    bottle=[i for i in range(len(pos)) if 0.01 < h[i] < 0.45]
    print(f"bottle pts = {len(bottle)}")
    # cluster in-plane
    cell=0.01; grid=defaultdict(list)
    for i in bottle:
        pu=dot(pos[i],u); pv=dot(pos[i],v)
        grid[(int(round(pu/cell)),int(round(pv/cell)))].append(i)
    seen=set(); comps=[]
    for c in grid:
        if c in seen: continue
        q=deque([c]); seen.add(c); cur=[c]
        while q:
            x,z=q.popleft()
            for dx,dz in ((1,0),(-1,0),(0,1),(0,-1)):
                nb=(x+dx,z+dz)
                if nb in grid and nb not in seen:
                    seen.add(nb); q.append(nb); cur.append(nb)
        comps.append(cur)
    comps.sort(key=lambda c:sum(len(grid[cc]) for cc in c), reverse=True)
    # union of top-2 comps (likely bottle split by plane cut)
    sel=[]
    for comp in comps[:2]:
        for cc in comp: sel.extend(grid[cc])
    print(f"merged top2 comps n={len(sel)}")
    # footprint in (u,v), height in h
    pus=[dot(pos[i],u) for i in sel]; pvs=[dot(pos[i],v) for i in sel]; hs=[h[i] for i in sel]
    fu=max(pus)-min(pus); fv=max(pvs)-min(pvs); fh=max(hs)-min(hs)
    print(f"footprint u={fu:.3f} v={fv:.3f}  height={fh:.3f}")
    real_long,real_wide,real_tall=0.277,0.197,0.283
    # assign longer footprint axis -> long, shorter -> wide, height->tall
    if fu>=fv:
        k_long=real_long/fu; k_wide=real_wide/fv
    else:
        k_long=real_long/fv; k_wide=real_wide/fu
    k_tall=real_tall/fh
    print(f"k_long={k_long:.3f} k_wide={k_wide:.3f} k_tall={k_tall:.3f}")
    print(f"mean k = {(k_long+k_wide+k_tall)/3:.3f}  => scene/real = {3/(k_long+k_wide+k_tall):.3f}x")

if __name__=='__main__':
    main()
