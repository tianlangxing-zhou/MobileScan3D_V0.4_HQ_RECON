#!/usr/bin/env python3
"""Host regression for round 2+3. g++, Python 3; no Android/OpenCV/Ceres."""
from pathlib import Path
import os,subprocess,tempfile,json,struct
root=Path(__file__).resolve().parents[2];cpp=root/'app/src/main/cpp'
with tempfile.TemporaryDirectory() as tmp:
 binary=Path(tmp)/'round23'
 stub=Path(tmp)/'android';stub.mkdir();(stub/'log.h').write_text('#pragma once\n#define ANDROID_LOG_WARN 5\ninline int __android_log_print(int, const char*, const char*, ...) {return 0;}\n')
 sources=['depth_calib.cpp','tsdf_engine.cpp','mesh/mesh_engine.cpp','mesh/hard_surface.cpp','export/gltf_exporter.cpp','v06/uv_unwrap.cpp','third_party/xatlas/xatlas.cpp']
 subprocess.run(['g++','-std=c++20','-O1','-g','-fsanitize=address,undefined,float-cast-overflow','-fno-sanitize-recover=all','-fno-omit-frame-pointer','-DMOBILESCAN_HAS_XATLAS=1','-pthread','-I',tmp,'-I',str(cpp/'third_party/xatlas'),'-I',str(cpp),str(Path(__file__).with_name('round23_regression.cpp')),*[str(cpp/p) for p in sources],'-o',str(binary)],check=True)
 subprocess.run([str(binary),tmp],env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0'),check=True)
 raw=(Path(tmp)/'cuboid.glb').read_bytes();magic,version,length=struct.unpack_from('<III',raw);assert magic==0x46546C67 and version==2 and length==len(raw)
 n=struct.unpack_from('<I',raw,12)[0];doc=json.loads(raw[20:20+n]);attrs=doc['meshes'][0]['primitives'][0]['attributes'];assert doc['accessors'][attrs['POSITION']]['count']==24;assert doc['accessors'][doc['meshes'][0]['primitives'][0]['indices']]['count']==36
 print('PASS cuboid GLB header, 24 vertices / 36 indices')
