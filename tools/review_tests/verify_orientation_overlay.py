#!/usr/bin/env python3
"""Read-only verification of the V0.13.17 overlay, in extracted ZIP or project root."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
manifest = json.loads((root/'ORIENTATION_OVERLAY_MANIFEST.json').read_text(encoding='utf-8'))
failed = []
for item in manifest['files']:
    path = root / item['path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        failed.append(item['path'])
if failed:
    raise SystemExit('FAIL missing or changed:\n' + '\n'.join(failed))
print(f"PASS {len(manifest['files'])} overlay files; baseline {manifest['base_commit']}")
