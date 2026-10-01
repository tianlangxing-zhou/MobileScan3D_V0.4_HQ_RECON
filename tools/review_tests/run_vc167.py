#!/usr/bin/env python3
"""VC167 host regression. Requires Python 3 and g++; no Android SDK required."""
from pathlib import Path
import subprocess, tempfile
root = Path(__file__).resolve().parents[2]
cpp = root / 'app/src/main/cpp'
with tempfile.TemporaryDirectory() as work:
    exe = Path(work) / 'vc167'
    subprocess.run(['g++', '-pipe', '-std=c++20', '-O1',
                    '-fsanitize=undefined,bounds,float-cast-overflow',
                    '-fno-sanitize-recover=all', '-I', str(cpp),
                    str(Path(__file__).with_name('vc167_regression.cpp')),
                    str(cpp / 'depth_calib.cpp'), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
