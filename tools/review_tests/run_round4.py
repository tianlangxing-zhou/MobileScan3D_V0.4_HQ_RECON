#!/usr/bin/env python3
"""Round 4 host regression; --measure builds optimized timing run (not Android)."""
from pathlib import Path
import argparse,os,subprocess,tempfile,statistics,re
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--measure',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];cpp=root/'app/src/main/cpp'
with tempfile.TemporaryDirectory() as tmp:
 binary=Path(tmp)/'round4'
 options=['-O3'] if a.measure else ['-O1','-g','-fsanitize=address,undefined,float-cast-overflow','-fno-sanitize-recover=all','-fno-omit-frame-pointer']
 subprocess.run(['g++','-std=c++20',*options,'-Wall','-Wextra','-I',str(cpp),str(Path(__file__).with_name('round4_regression.cpp')),*[str(cpp/s) for s in ('surfel_engine.cpp','tsdf_engine.cpp','mesh/mesh_engine.cpp')],'-o',str(binary)],check=True)
 if a.measure:
  dense=[];adaptive=[]
  for i in range(5):
   result=subprocess.run([str(binary),'--measure'],check=True,capture_output=True,text=True)
   print(f'REPEAT {i+1}\n'+result.stdout,flush=True)
   dense.append(float(re.search(r'dense_ms_per_frame=([\d.]+)',result.stdout)[1]))
   adaptive.append(float(re.search(r'adaptive_ms_per_frame=([\d.]+)',result.stdout)[1]))
  a_ms=statistics.median(dense);b_ms=statistics.median(adaptive)
  print(f'host_median_dense_ms={a_ms:.6f} host_median_adaptive_ms={b_ms:.6f} reduction_percent={(1-b_ms/a_ms)*100:.2f}')
 else:
  subprocess.run([str(binary)],check=True,env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0'))
