#!/usr/bin/env python3
"""Native contour/overlap regressions; --measure times 256-square inputs on host."""
from pathlib import Path
import argparse,subprocess,tempfile
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--measure',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parents[2];cpp=root/'app/src/main/cpp'
with tempfile.TemporaryDirectory() as tmp:
    binary=Path(tmp)/'contour_guard'
    options=['-O3'] if args.measure else ['-O1','-g','-fsanitize=address,undefined,float-cast-overflow','-fno-sanitize-recover=all','-fno-omit-frame-pointer']
    source='contour_guard_benchmark.cpp' if args.measure else 'contour_guard_regression.cpp'
    deps=['depth_calib.cpp'] if args.measure else ['surfel_engine.cpp','depth_calib.cpp','tsdf_engine.cpp','mesh/mesh_engine.cpp']
    subprocess.run(['g++','-std=c++20',*options,'-Wall','-Wextra','-I',str(cpp),
        str(Path(__file__).with_name(source)),*[str(cpp/s) for s in deps],'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
