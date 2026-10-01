#!/usr/bin/env python3
import struct, json
base='E:/MobileScan3D/devtest/analyze_vc168/scan_20261001_145835_15281'
with open(base+'.glb','rb') as f:
    data=f.read()
magic,ver,length=struct.unpack_from('<III', data, 0)
off=12
clen,ctype=struct.unpack_from('<II', data, off)
gltf=json.loads(data[off+8:off+8+clen].decode('utf-8'))
print("=== SCENES ===")
print(json.dumps(gltf.get('scenes'), indent=1)[:1500])
print("=== NODES (transform-bearing) ===")
for i,n in enumerate(gltf.get('nodes',[])):
    name=n.get('name','')
    keys={k:n[k] for k in ('matrix','translation','rotation','scale') if k in n}
    mesh=n.get('mesh')
    print(f"node[{i}] name={name!r} mesh={mesh} transform={keys}")
print("=== MESH prim POSITION accessor min/max ===")
for mi,m in enumerate(gltf.get('meshes',[])):
    for pi,prim in enumerate(m['primitives']):
        pid=prim['attributes']['POSITION']
        acc=gltf['accessors'][pid]
        print(f"mesh[{mi}] prim[{pi}] POSITION acc#{pid} count={acc['count']} min={acc.get('min')} max={acc.get('max')}")
        # also check if indices exist
        print(f"   hasIndices={'indices' in prim} mode={prim.get('mode')}")
print("=== top-level asset/generator ===")
print(gltf.get('asset'))
# bufferView byteLength for POSITION
for mi,m in enumerate(gltf.get('meshes',[])):
    for pi,prim in enumerate(m['primitives']):
        pid=prim['attributes']['POSITION']
        acc=gltf['accessors'][pid]
        bv=gltf['bufferViews'][acc['bufferView']]
        print(f"mesh[{mi}] prim[{pi}] bufferView byteLength={bv.get('byteLength')} byteOffset={bv.get('byteOffset')} byteStride={bv.get('byteStride')}")
