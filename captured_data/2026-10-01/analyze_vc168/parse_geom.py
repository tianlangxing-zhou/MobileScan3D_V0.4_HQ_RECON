#!/usr/bin/env python3
import struct, json, math, sys

def parse_ply_ascii(path):
    xs=[]; ys=[]; zs=[]
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        # read header
        nvert=0
        while True:
            line=f.readline()
            if not line: break
            s=line.strip()
            if s=='end_header': break
            if s.startswith('element vertex'):
                nvert=int(s.split()[2])
        for _ in range(nvert):
            parts=f.readline().split()
            if len(parts)<3: continue
            xs.append(float(parts[0])); ys.append(float(parts[1])); zs.append(float(parts[2]))
    return xs,ys,zs

def parse_glb(path):
    with open(path,'rb') as f:
        data=f.read()
    magic,ver,length=struct.unpack_from('<III', data, 0)
    assert magic==0x46546C67, 'not glTF'
    # chunk 0 JSON
    off=12
    clen,ctype=struct.unpack_from('<II', data, off)
    jchunk=data[off+8:off+8+clen]
    gltf=json.loads(jchunk.decode('utf-8'))
    off+=8+clen
    # chunk 1 BIN
    blen,btype=struct.unpack_from('<II', data, off)
    bin_start=off+8
    bin_data=data[bin_start:bin_start+blen]
    # find POSITION accessor of first mesh primitive
    meshes=gltf.get('meshes',[])
    prim=meshes[0]['primitives'][0]
    pos_acc_id=prim['attributes']['POSITION']
    acc=gltf['accessors'][pos_acc_id]
    bv=gltf['bufferViews'][acc['bufferView']]
    comp=acc['componentType']  # 5126 FLOAT
    count=acc['count']
    aoff=bv.get('byteOffset',0)+acc.get('byteOffset',0)
    stride=bv.get('byteStride',0) or 12
    assert comp==5126
    xs=[];ys=[];zs=[]
    for i in range(count):
        base=bin_start+bv.get('byteOffset',0)+i*stride
        x,y,z=struct.unpack_from('<fff', bin_data, base)
        xs.append(x);ys.append(y);zs.append(z)
    return xs,ys,zs

def stats(name, xs,ys,zs):
    n=len(xs)
    if n==0:
        print(f"[{name}] EMPTY"); return
    minx,maxx=min(xs),max(xs)
    miny,maxy=min(ys),max(ys)
    minz,maxz=min(zs),max(zs)
    sx=maxx-minx; sy=maxy-miny; sz=maxz-minz
    # centroid
    cx=sum(xs)/n; cy=sum(ys)/n; cz=sum(zs)/n
    # PCA on centered coords for principal extent
    # build covariance (3x3)
    sxx=syy=szz=sxy=sxz=syz=0.0
    for i in range(n):
        dx=xs[i]-cx; dy=ys[i]-cy; dz=zs[i]-cz
        sxx+=dx*dx; syy+=dy*dy; szz+=dz*dz
        sxy+=dx*dy; sxz+=dx*dz; syz+=dy*dz
    sxx/=n; syy/=n; szz/=n; sxy/=n; sxz/=n; syz/=n
    # eigen of symmetric 3x3 via numpy-free Jacobi
    A=[[sxx,sxy,sxz],[sxy,syy,syz],[sxz,syz,szz]]
    # power iteration for largest eigenvalue/vector
    v=[1.0,0.0,0.0]
    for _ in range(100):
        nv=[A[0][0]*v[0]+A[0][1]*v[1]+A[0][2]*v[2],
            A[1][0]*v[0]+A[1][1]*v[1]+A[1][2]*v[2],
            A[2][0]*v[0]+A[2][1]*v[1]+A[2][2]*v[2]]
        norm=math.sqrt(nv[0]**2+nv[1]**2+nv[2]**2)+1e-12
        v=[x/norm for x in nv]
    lam=v[0]*(A[0][0]*v[0]+A[0][1]*v[1]+A[0][2]*v[2])+v[1]*(A[1][0]*v[0]+A[1][1]*v[1]+A[1][2]*v[2])+v[2]*(A[2][0]*v[0]+A[2][1]*v[1]+A[2][2]*v[2])
    principal_extent=2*math.sqrt(max(lam,0.0)*n)
    print(f"[{name}] n={n}")
    print(f"  AABB  X=[{minx:.4f},{maxx:.4f}] span={sx:.4f}")
    print(f"  AABB  Y=[{miny:.4f},{maxy:.4f}] span={sy:.4f}")
    print(f"  AABB  Z=[{minz:.4f},{maxz:.4f}] span={sz:.4f}")
    print(f"  diag(AABB)={math.sqrt(sx*sx+sy*sy+sz*sz):.4f}")
    print(f"  centroid=({cx:.4f},{cy:.4f},{cz:.4f})")
    print(f"  principal(axis-aligned longest)={max(sx,sy,sz):.4f}")
    print(f"  principal(PCA longest)={principal_extent:.4f}")

if __name__=='__main__':
    base='E:/MobileScan3D/devtest/analyze_vc168/scan_20261001_145835_15281'
    ply=base+'_debug.ply'
    glb=base+'.glb'
    print("=== DEBUG POINT CLOUD (PLY) ===")
    stats('cloud', *parse_ply_ascii(ply))
    print()
    print("=== BAKED MESH (GLB) ===")
    stats('mesh', *parse_glb(glb))
