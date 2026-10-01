import json, struct, sys, re

def read_glb_accessor_bounds(path):
    with open(path, 'rb') as f:
        magic, ver, length = struct.unpack('<III', f.read(12))
        assert magic == 0x46546C67, 'not glb'
        # JSON chunk
        clen, ctype = struct.unpack('<II', f.read(8))
        js = f.read(clen).decode('utf-8')
    g = json.loads(js)
    acc = g['accessors']
    # find POSITION accessor of first mesh primitive
    mesh = g['meshes'][0]
    pos_idx = mesh['primitives'][0]['attributes']['POSITION']
    a = acc[pos_idx]
    return a.get('min'), a.get('max'), len(acc)

def read_ply_bounds(path):
    mn = [ float('inf') ] * 3
    mx = [ float('-inf') ] * 3
    n = 0
    with open(path, 'r') as f:
        # header
        line = f.readline()
        assert line.strip() == 'ply'
        count = 0
        while True:
            line = f.readline()
            if line.startswith('element vertex'):
                count = int(line.split()[2])
            if line.strip() == 'end_header':
                break
        for _ in range(count):
            parts = f.readline().split()
            if len(parts) < 3:
                continue
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            n += 1
            for i, v in enumerate((x, y, z)):
                if v < mn[i]: mn[i] = v
                if v > mx[i]: mx[i] = v
    return mn, mx, n

def spans(mn, mx):
    return [round(mx[i]-mn[i], 4) for i in range(3)]

def main():
    base = sys.argv[1]
    glb = base + '.glb'
    ply = base + '_debug.ply'
    gmn, gmx, nacc = read_glb_accessor_bounds(glb)
    print('=== GLB accessor bounds (authoritative world coords) ===')
    print('  min =', [round(v,4) for v in gmn])
    print('  max =', [round(v,4) for v in gmx])
    gs = spans(gmn, gmx)
    print('  span X,Y,Z =', gs, ' diag =', round(sum(s*s for s in gs)**0.5,4))
    print('  axis-sorted spans (desc):', sorted(gs, reverse=True))

    pmn, pmx, pn = read_ply_bounds(ply)
    print('=== PLY cloud bounds (%d pts) ===' % pn)
    print('  min =', [round(v,4) for v in pmn])
    print('  max =', [round(v,4) for v in pmx])
    ps = spans(pmn, pmx)
    print('  span X,Y,Z =', ps, ' diag =', round(sum(s*s for s in ps)**0.5,4))
    print('  axis-sorted spans (desc):', sorted(ps, reverse=True))

    # ground truth Dettol bottle: 长277 宽197 高283 mm
    real = [0.277, 0.197, 0.283]  # 长,宽,高
    real_sorted = sorted(real, reverse=True)
    print('=== ground truth (sorted desc) mm ===')
    print('  real =', [round(v*1000) for v in real_sorted], 'mm')
    print('=== scale k per axis (real / scene-span), sorted-match ===')
    gs_sorted = sorted(gs, reverse=True)
    for i, (r, s) in enumerate(zip(real_sorted, gs_sorted)):
        k = r/s if s > 1e-6 else float('nan')
        print('  axis%d: real=%.3f  scene=%.3f  k=%.3f' % (i, r, s, k))
    kmean = sum(real_sorted[i]/gs_sorted[i] for i in range(3) if gs_sorted[i]>1e-6)/3
    print('  mean k (sorted match) =', round(kmean,3))
    # also raw per-axis vs unsorted real (长=x? 高=y? unknown) just report
    print('=== if scene axes map to real 长/宽/高 in order X,Y,Z ===')
    for ax, (rn, rv, sv) in enumerate(zip(['长','宽','高'], real, gs)):
        print('  %s: real=%.3f scene=%.3f k=%.3f' % (rn, rv, sv, rv/sv if sv>1e-6 else float('nan')))

if __name__ == '__main__':
    main()
