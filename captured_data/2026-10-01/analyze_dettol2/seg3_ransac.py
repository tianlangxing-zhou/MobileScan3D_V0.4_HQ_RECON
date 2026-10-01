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
    a=g['accessors'][0]
    bv=g['bufferViews'][a['bufferView']]
    off=bv.get('byteOffset',0)+a.get('byteOffset',0)
    cnt=a['count']
    pos=[]
    for i in range(cnt):
        base=off+i*stride
        x,y,z=struct.unpack_from('<fff', bin_data, base)
        pos.append((x,y,z))
    return pos

def ransac_plane(pts, iters=500, eps=0.012):
    n=len(pts)
    best=None; best_in=0
    for _ in range(iters):
        i,j,k=random.sample(range(n),3)
        p1,p2,p3=pts[i],pts[j],pts[k]
        ax,ay,az=p2[0]-p1[0],p2[1]-p1[1],p2[2]-p1[2]
        bx,by,bz=p3[0]-p1[0],p3[1]-p1[1],p3[2]-p1[2]
        nx=ay*bz-az*by; ny=az*bx-ax*bz; nz=ax*by-ay*bx
        ln=math.sqrt(nx*nx+ny*ny+nz*nz)
        if ln<1e-9: continue
        nx,ny,nz=nx/ln,ny/ln,nz/ln
        d=-(nx*p1[0]+ny*p1[1]+nz*p1[2])
        # signed dist
        inn=0
        for p in pts:
            dist=abs(nx*p[0]+ny*p[1]+nz*p[2]+d)
            if dist<eps: inn+=1
        if inn>best_in:
            best_in=inn; best=(nx,ny,nz,d)
    return best,best_in

def main():
    base=sys.argv[1]
    glb=base+'.glb'
    pos=extract_positions_strided(glb,32)
    print(f"verts={len(pos)}")
    # subsample for ransac
    step=max(1,len(pos)//40000)
    sub=pos[::step]
    print(f"ransac subsample={len(sub)}")
    plane,inl=ransac_plane(sub, iters=600, eps=0.012)
    nx,ny,nz,d=plane
    print(f"plane normal=({nx:.3f},{ny:.3f},{nz:.3f}) d={d:.3f} inliers={inl}/{len(sub)} ({100*inl/len(sub):.1f}%)")
    # orient normal so that most points are on +side and 'up' positive y
    # height above plane
    h=[nx*p[0]+ny*p[1]+nz*p[2]+d for p in pos]
    # ensure bottle (the tall object) is on + side: flip if needed
    # bottle pts = h in (0.02, 0.45)
    bottle=[i for i in range(len(pos)) if 0.02 < h[i] < 0.45]
    print(f"bottle-candidate pts (0.02<h<0.45) = {len(bottle)}")
    if len(bottle)<50:
        print("try widen: 0.0<h<0.6")
        bottle=[i for i in range(len(pos)) if 0.0 < h[i] < 0.6]
        print(f"  ={len(bottle)}")
    # cluster in XZ (table approx horizontal; use x,z)
    cell=0.01
    grid=defaultdict(list)
    for i in bottle:
        gx=int(round(pos[i][0]/cell)); gz=int(round(pos[i][2]/cell))
        grid[(gx,gz)].append(i)
    # connected comps 4-neigh
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
    print(f"comps={len(comps)} sizes(top5)={[sum(len(grid[cc]) for cc in c) for c in comps[:5]]}")
    real=[0.283,0.277,0.197]
    for ci,comp in enumerate(comps[:3]):
        sel=[i for cc in comp for i in grid[cc]]
        if len(sel)<20: continue
        xs=[pos[i][0] for i in sel]; ys=[pos[i][1] for i in sel]; zs=[pos[i][2] for i in sel]
        span=(max(xs)-min(xs),max(ys)-min(ys),max(zs)-min(zs))
        order=sorted(range(3),key=lambda a:-span[a])
        ks=[real[r]/span[order[r]] for r in range(3)]
        print(f"  comp#{ci} n={len(sel)} spanX={span[0]:.3f} Y={span[1]:.3f} Z={span[2]:.3f} k(sorted)={[round(k,3) for k in ks]} mean={sum(ks)/3:.3f}")

if __name__=='__main__':
    main()
