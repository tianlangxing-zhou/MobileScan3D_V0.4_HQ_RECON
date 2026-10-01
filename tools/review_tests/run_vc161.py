#!/usr/bin/env python3
"""VC161 regressions. Python 3 + g++; --opencv adds desktop OpenCV tests.
OpenCV tracker test needs TrackerNano declarations (OpenCV >=4.7); no model is loaded.
Set KOTLINC/JAVA to additionally run the Kotlin buffer numerical regression.
"""
from pathlib import Path
import argparse, os, shlex, shutil, subprocess, tempfile
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--opencv', action='store_true')
a=p.parse_args()
root=Path(__file__).resolve().parents[2];cpp=root/'app/src/main/cpp';tests=Path(__file__).parent
with tempfile.TemporaryDirectory() as tmp:
    binary=Path(tmp)/'depth'
    subprocess.run(['g++','-std=c++20','-O2','-fsanitize=undefined,bounds,float-cast-overflow',
        '-fno-sanitize-recover=all','-I',str(cpp),str(tests/'vc161_depth_regression.cpp'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
    if a.opencv:
        flags=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','opencv4'],text=True))
        for name in ['reacquire','tracker']:
            deps=[] if name=='reacquire' else [str(cpp/'object_tracker.cpp'),str(cpp/'appearance_tracker.cpp')]
            subprocess.run(['g++','-std=c++20','-O2','-I',str(cpp),str(tests/f'vc161_{name}_regression.cpp'),*deps,*flags,'-o',str(binary)],check=True)
            subprocess.run([str(binary)],check=True)
    compiler=os.environ.get('KOTLINC') or shutil.which('kotlinc')
    if compiler:
        jar=Path(tmp)/'buffer.jar'
        subprocess.run([compiler,str(root/'app/src/main/java/com/mobilescan3d/depth/DepthPreprocessor.kt'),
            str(tests/'DepthBufferRegression.kt'),'-include-runtime','-d',str(jar)],check=True)
        subprocess.run([os.environ.get('JAVA') or 'java','-jar',str(jar)],check=True)
    else: print('SKIP Kotlin buffer regression: set KOTLINC to enable')
