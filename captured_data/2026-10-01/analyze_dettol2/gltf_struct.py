import json, struct, sys

def main():
    path=sys.argv[1]
    with open(path,'rb') as f:
        magic,ver,length=struct.unpack('<III',f.read(12))
        clen,ctype=struct.unpack('<II',f.read(8))
        js=f.read(clen).decode('utf-8')
    g=json.loads(js)
    print("=== meshes ===")
    for i,m in enumerate(g.get('meshes',[])):
        print(f" mesh[{i}] name={m.get('name')} prims={len(m['primitives'])}")
        for j,p in enumerate(m['primitives']):
            a=p['attributes'].get('POSITION')
            acc=g['accessors'][a]
            print(f"   prim[{j}] POSITION accessor={a} count={acc['count']} min={[round(x,3) for x in acc.get('min')]} max={[round(x,3) for x in acc.get('max')]}")
    print("=== nodes ===")
    for i,n in enumerate(g.get('nodes',[])):
        print(f" node[{i}] name={n.get('name')} mesh={n.get('mesh')} matrix={[round(x,4) for x in n['matrix']] if 'matrix' in n else n.get('translation')} scale={n.get('scale')} rot={n.get('rotation')}")
    print("=== scenes ===", g.get('scenes'))
    print("=== asset ===", g.get('asset'))
    print("=== bufferViews[0..3] ===")
    for i,bv in enumerate(g.get('bufferViews',[])[:4]):
        print(f"  bv[{i}] buffer={bv.get('buffer')} byteOffset={bv.get('byteOffset')} byteLength={bv.get('byteLength')} target={bv.get('target')} byteStride={bv.get('byteStride')}")
    a0=g['accessors'][0]
    print("=== accessor[0] ===", {k:a0.get(k) for k in ('bufferView','byteOffset','count','componentType','type')})
    # total vertex count first primitive only
    if g['meshes']:
        p=g['meshes'][0]['primitives'][0]
        a=p['attributes']['POSITION']
        print("first prim POSITION count =", g['accessors'][a]['count'])

if __name__=='__main__':
    main()
