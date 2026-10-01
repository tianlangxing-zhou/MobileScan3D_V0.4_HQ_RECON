#!/usr/bin/env python3
"""Verify the delivered VC167 files after overlaying them onto the project."""
from pathlib import Path
import hashlib, json, sys
root = Path(__file__).resolve().parents[2]
manifest = json.loads((root / 'VC167_MANIFEST.json').read_text(encoding='utf-8'))
errors = []
for item in manifest['files']:
    path = root / item['path']
    if not path.is_file():
        errors.append('MISSING ' + item['path'])
    elif hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        errors.append('MISMATCH ' + item['path'])
if errors:
    print('\n'.join(errors)); sys.exit(1)
print('PASS VC167 overlay: %d files match SHA-256' % len(manifest['files']))
