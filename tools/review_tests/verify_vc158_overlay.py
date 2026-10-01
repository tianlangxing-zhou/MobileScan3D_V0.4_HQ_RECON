#!/usr/bin/env python3
from pathlib import Path
import argparse,hashlib,json,sys
p=argparse.ArgumentParser(description='Verify original vc157 baseline or installed vc158 patch')
p.add_argument('--before',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[2]
m=json.loads((root/'docs/vc158/manifest.json').read_text())
base=a.before.resolve() if a.before else root
errors=0
for item in m['files']:
    path=base/item['path'];expected=item['before_sha256'] if a.before else item['after_sha256']
    actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if actual!=expected:print('DIFFERENT or MISSING',item['path']);errors+=1
print('PASS' if not errors else f'FAIL: {errors} files differ')
sys.exit(bool(errors))
