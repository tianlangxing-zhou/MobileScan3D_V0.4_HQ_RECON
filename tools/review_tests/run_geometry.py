#!/usr/bin/env python3
"""Native geometry regression with address/undefined behavior sanitizers."""
from pathlib import Path
import os, subprocess, tempfile
root=Path(__file__).resolve().parents[2]
cpp=root/'app/src/main/cpp'
with tempfile.TemporaryDirectory() as work:
    binary=Path(work)/'geometry'
    sources=['depth_calib.cpp','tsdf_engine.cpp','surfel_engine.cpp','mesh/mesh_engine.cpp']
    subprocess.run(['g++','-std=c++20','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer',
                    '-I',str(cpp),str(Path(__file__).with_name('geometry_regression.cpp')),
                    *[str(cpp/s) for s in sources],'-o',str(binary)],check=True)
    subprocess.run([str(binary)],env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0'),check=True)
