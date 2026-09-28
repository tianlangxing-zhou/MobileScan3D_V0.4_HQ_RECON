from pathlib import Path
import os, subprocess, tempfile
root=Path(__file__).resolve().parents[2]
cpp=root/'app/src/main/cpp'
with tempfile.TemporaryDirectory() as work:
    binary=Path(work)/'scan_policy'
    subprocess.run(['g++','-std=c++20','-O1','-g','-fsanitize=address,undefined',
        '-fno-omit-frame-pointer','-I',str(cpp),str(Path(__file__).with_name('scan_policy_regression.cpp')),
        str(cpp/'tsdf_engine.cpp'),str(cpp/'mesh/mesh_engine.cpp'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0'),check=True)
