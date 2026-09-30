#!/usr/bin/env python3
"""Read-only verification of this incremental patch before/after local overlay."""
from pathlib import Path
import argparse,hashlib,json,sys
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--before',action='store_true',help='accept original baseline or already patched files; allow absent new files')
parser.add_argument('project',type=Path,nargs='?',default=Path(__file__).resolve().parents[2])
a=parser.parse_args()
manifest=Path(__file__).resolve().parents[2]/'docs/round23/manifest.json'
rows=json.loads(manifest.read_text(encoding='utf-8'))['files'];errors=[]
for row in rows:
 p=a.project/row['path']
 if not p.is_file():
  if a.before and row['baseline_sha256'] is None:continue
  errors.append('MISSING '+row['path']);continue
 digest=hashlib.sha256(p.read_bytes()).hexdigest()
 allowed={row['sha256']}
 if a.before and row['baseline_sha256']:allowed.add(row['baseline_sha256'])
 if digest not in allowed:errors.append('DIFFERENT '+row['path'])
for x in errors:print(x)
print(('FAIL' if errors else 'PASS')+f' {len(rows)} patch files checked')
sys.exit(bool(errors))
