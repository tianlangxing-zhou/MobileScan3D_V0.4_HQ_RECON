#!/usr/bin/env python3
"""Read-only verification of delivered files after merging the overlay."""
from pathlib import Path
import hashlib
import json
import sys

root = Path(__file__).resolve().parents[2]
manifest = root / 'OVERLAY_MANIFEST.json'
if not manifest.is_file():
    sys.exit('Missing OVERLAY_MANIFEST.json: copy it from the ZIP to the project root.')
data = json.loads(manifest.read_text(encoding='utf-8'))
failed = []
for item in data['files']:
    path = root / item['path']
    if not path.is_file():
        failed.append('MISSING ' + item['path'])
    elif hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        failed.append('CHANGED ' + item['path'])
if failed:
    print('\n'.join(failed))
    sys.exit(1)
print(f"PASS {len(data['files'])} delivered files; base {data['base_commit']}")
