#!/usr/bin/env python3
"""Verify this patch's exact uploaded baseline or the installed replacement files."""
from pathlib import Path
import argparse, hashlib, json, sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--before',type=Path,help='Original vc155 project root before merging the patch')
a=p.parse_args()
root=Path(__file__).resolve().parents[2]
manifest=json.loads((root/'docs/fusion_refinement/manifest.json').read_text())
target=a.before.resolve() if a.before else root
errors=0
for item in manifest['files']:
    expected=item['before_sha256'] if a.before else item['after_sha256']
    path=target/item['path']
    if expected is None:
        if path.exists():
            print('CONFLICT (new patch file already exists)',item['path']);errors+=1
        continue
    actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if actual!=expected:
        print('DIFFERENT or MISSING',item['path']);errors+=1
print('PASS' if errors==0 else f'FAIL: {errors} files differ')
sys.exit(bool(errors))
