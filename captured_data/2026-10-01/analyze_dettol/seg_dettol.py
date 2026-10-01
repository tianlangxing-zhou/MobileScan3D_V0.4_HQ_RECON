import sys

def load_ply(path):
    pts = []
    with open(path, 'r') as f:
        assert f.readline().strip() == 'ply'
        count = 0
        while True:
            line = f.readline()
            if line.startswith('element vertex'):
                count = int(line.split()[2])
            if line.strip() == 'end_header':
                break
        for _ in range(count):
            p = f.readline().split()
            if len(p) < 3: continue
            pts.append((float(p[0]), float(p[1]), float(p[2])))
    return pts

def main():
    base = sys.argv[1]
    pts = load_ply(base + '_debug.ply')
    n = len(pts)
    ys = sorted(p[1] for p in pts)
    xs = [p[0] for p in pts]
    zs = [p[2] for p in pts]
    import statistics
    def pct(a, q):
        return a[int(q*(len(a)-1))]
    print('total pts =', n)
    print('Y  min/med/max = %.3f / %.3f / %.3f' % (ys[0], ys[n//2], ys[-1]))
    print('Y  p05/p50/p95 = %.3f / %.3f / %.3f' % (pct(ys,.05), pct(ys,.5), pct(ys,.95)))
    # histogram of Y to find desk plane peak in lower region
    nb = 60
    lo, hi = ys[0], ys[-1]
    bins = [0]*(nb+1)
    for y in ys:
        b = int((y-lo)/(hi-lo+1e-9)*nb)
        bins[min(b,nb)] += 1
    peak = max(range(nb+1), key=lambda i: bins[i])
    deskY = lo + (peak+0.5)/nb*(hi-lo)
    print('Y histogram peak (desk candidate) bin=%d deskY=%.3f' % (peak, deskY))
    # bottle = points clearly above desk
    above = [p for p in pts if p[1] > deskY + 0.03]
    if above:
        ax = [p[0] for p in above]; ay=[p[1] for p in above]; az=[p[2] for p in above]
        print('above-desk pts =', len(above))
        print('  bottle X span = %.3f (%.3f..%.3f)' % (max(ax)-min(ax), min(ax), max(ax)))
        print('  bottle Y span = %.3f (%.3f..%.3f)  [height above desk]' % (max(ay)-min(ay), min(ay), max(ay)))
        print('  bottle Z span = %.3f (%.3f..%.3f)' % (max(az)-min(az), min(az), max(az)))
    # horizontal centroid of above-desk, keep compact cluster within 0.25
    if above:
        cx = sum(p[0] for p in above)/len(above); cz = sum(p[2] for p in above)/len(above)
        core = [p for p in above if abs(p[0]-cx)<0.22 and abs(p[2]-cz)<0.22]
        if core:
            cxy=[p[1] for p in core]; cxx=[p[0] for p in core]; czz=[p[2] for p in core]
            print('core-cluster pts =', len(core))
            print('  core X span=%.3f Y span=%.3f Z span=%.3f' % (max(cxx)-min(cxx), max(cxy)-min(cxy), max(czz)-min(czz)))

if __name__ == '__main__':
    main()
