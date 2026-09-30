#!/usr/bin/env python3
"""Production geometry/asset boundary regressions; g++ and Python only."""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
cpp = root / 'app/src/main/cpp'
with tempfile.TemporaryDirectory() as work:
    binary = Path(work) / 'robustness'
    subprocess.run(['g++', '-std=c++20', '-O1', '-g',
        '-fsanitize=address,undefined,float-cast-overflow',
        '-fno-sanitize-recover=all', '-fno-omit-frame-pointer', '-I', str(cpp),
        str(Path(__file__).with_name('robustness_regression.cpp')),
        *[str(cpp / name) for name in ['tsdf_engine.cpp',
            'mesh/mesh_engine.cpp', 'ar_textured_asset.cpp']], '-o', str(binary)], check=True)
    env = dict(os.environ, ASAN_OPTIONS='detect_leaks=0')
    failed = []
    for mode in ('numeric', 'ray', 'cluster', 'asset'):
        result = subprocess.run([str(binary), mode, work], env=env)
        if result.returncode:
            failed.append(mode)
    if failed:
        raise SystemExit('FAILED: ' + ', '.join(failed))
