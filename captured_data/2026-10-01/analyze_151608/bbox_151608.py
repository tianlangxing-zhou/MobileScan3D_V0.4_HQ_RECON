#!/usr/bin/env python3
import struct, json, sys, math

def glb_bbox(path):
    with open(path,'rb') as f: data=f.read()
    magic,ver,length=struct.unpack_from('<III', data, 0)
    off=12; clen,ctype=struct.unpack_from('<II', data, off)
    gltf=json.loads(data[off+8:off+8+clen].decode('utf-8'))
    m=gltf['meshes'][0]['primitives'][0]
    pid=m['attributes']['POSITION']
    acc=gltf['accessors'][pid]
    n=acc['count']
    mn=acc.get('min'); mx=acc.get('max')
    print(f"[GLB] POSITION count={n}")
    if mn and mx:
        print(f"  min=({mn[0]:.4f},{mn[1]:.4f},{mn[2]:.4f})")
        print(f"  max=({mx[0]:.4f},{mx[1]:.4f},{mx[2]:.4f})")
        sx=mx[0]-mn[0]; sy=mx[1]-mn[1]; sz=mx[2]-mn[2]
        print(f"  span X={sx:.4f} Y={sy:.4f} Z={sz:.4f} diag={math.sqrt(sx*sx+sy*sy+sz*sz):.4f}")

def ply_bbox(path):
    xs=ys=zs=None
    xs=[];ys=[];zs=[]
    with open(path,'r',encoding='utf-8',errors='replace') as f:
        n=0
        while True:
            line=f.readline()
            if not line: break
            if line.strip()=='end_header': break
            if line.strip().startswith('element vertex'): n=int(line.split()[2])
        for _ in range(n):
            p=f.readline().split()
            if len(p)<3: continue
            xs.append(float(p[0]));ys.append(float(p[1]));zs.append(float(p[2]))
    if xs:
        mn=[min(xs),min(ys),min(zs)]; mx=[max(xs),max(ys),max(zs)]
        sx=mx[0]-mn[0];sy=mx[1]-mn[1];sz=mx[2]-mn[2]
        print(f"[PLY cloud] n={len(xs)}")
        print(f"  min=({mn[0]:.4f},{mn[1]:.4f},{mn[2]:.4f}) max=({mx[0]:.4f},{mx[1]:.4f},{mx[2]:.4f})")
        print(f"  span X={sx:.4f} Y={sy:.4f} Z={sz:.4f} diag={math.sqrt(sx*sx+sy*sy+sz*sz):.4f}")

base=sys.argv[1]
print("=== 151608 ===")
glb_bbox(base+'.glb')
ply_bbox(base+'_debug.ply')
