#!/usr/bin/env python3
"""Host correctness + optional before/after measurements (not phone benchmarks)."""
from pathlib import Path
import argparse, os, subprocess, tempfile
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline',type=Path)
a=p.parse_args()
root=Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as tmp:
    def build(project,name,sanitize):
        cpp=project/'app/src/main/cpp'
        options=['-O1','-g','-fsanitize=address,undefined,float-cast-overflow','-fno-sanitize-recover=all','-fno-omit-frame-pointer'] if sanitize else ['-O3']
        binary=Path(tmp)/name
        subprocess.run(['g++','-std=c++20',*options,'-Wall','-Wextra','-I',str(cpp),str(Path(__file__).with_name('fusion_refinement_regression.cpp')),str(cpp/'surfel_engine.cpp'),str(cpp/'depth_calib.cpp'),'-o',str(binary)],check=True)
        return binary
    subprocess.run([str(build(root,'check',True))],check=True,env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0'))
    if a.baseline:
        for name,project in [('baseline',a.baseline.resolve()),('modified',root)]:
            binary=build(project,name,False)
            for repeat in range(3):
                print(name,'repeat',repeat+1,flush=True)
                subprocess.run([str(binary),'--measure'],check=True)
