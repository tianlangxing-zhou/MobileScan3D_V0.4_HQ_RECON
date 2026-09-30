#!/usr/bin/env python3
"""Read-only before/after overlay validation for uploaded vc154 round23 baseline."""
from pathlib import Path
import argparse,hashlib,json,sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--before',action='store_true',help='accept original vc154 baseline or patched hashes')
p.add_argument('project',type=Path,nargs='?',default=Path(__file__).resolve().parents[2])
a=p.parse_args();root=Path(__file__).resolve().parents[2]
m=json.loads((root/'docs/round4/manifest.json').read_text(encoding='utf-8'));errors=[]
for row in m['files']+m['required_round23_files']:
 path=a.project/row['path']
 if not path.is_file():
  if a.before and row.get('baseline_sha256') is None and row in m['files']:continue
  errors.append('MISSING '+row['path']);continue
 allowed={row['sha256']}
 if a.before and row.get('baseline_sha256'):allowed.add(row['baseline_sha256'])
 if hashlib.sha256(path.read_bytes()).hexdigest() not in allowed:errors.append('DIFFERENT '+row['path'])
for line in errors:print(line)
print(('FAIL' if errors else 'PASS')+f" {len(m['files'])} changed/new files, {len(m['required_round23_files'])} retained round23 files")
sys.exit(bool(errors))
