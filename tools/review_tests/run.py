#!/usr/bin/env python3
"""Host regression tests, no Android/OpenCV/Ceres required. Requires g++, Pillow."""
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CPP = ROOT / 'app/src/main/cpp'


def inspect(path):
    raw = path.read_bytes()
    magic, version, length = struct.unpack_from('<III', raw)
    assert (magic, version, length) == (0x46546C67, 2, len(raw))
    n, kind = struct.unpack_from('<II', raw, 12)
    assert kind == 0x4E4F534A and n % 4 == 0
    doc = json.loads(raw[20:20+n])
    size, kind = struct.unpack_from('<II', raw, 20+n)
    assert kind == 0x004E4942 and size % 4 == 0
    binary = raw[28+n:]
    assert len(binary) == size == doc['buffers'][0]['byteLength']
    for v in doc['bufferViews']:
        assert v.get('byteOffset', 0) % 4 == 0
        assert v.get('byteOffset', 0) + v['byteLength'] <= len(binary)
    primitive = doc['meshes'][0]['primitives'][0]
    attrs = primitive['attributes']
    assert len(set(attrs.values())) == len(attrs)
    accessors = doc['accessors']
    for semantic, index in attrs.items():
        a = accessors[index]
        assert a['type'] == ('VEC2' if semantic == 'TEXCOORD_0' else
                             'VEC4' if semantic == 'COLOR_0' else 'VEC3')
        assert a['count'] == 3
    pos = accessors[attrs['POSITION']]
    view = doc['bufferViews'][pos['bufferView']]
    base = view.get('byteOffset', 0) + pos.get('byteOffset', 0)
    positions = [struct.unpack_from('<fff', binary, base+i*view.get('byteStride', 12)) for i in range(3)]
    for k in range(3):
        assert all(math.isfinite(p[k]) for p in positions)
        # Round-trip JSON bounds back to float32: bounds must match actual positions.
        f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
        assert f32(pos['min'][k]) == min(p[k] for p in positions)
        assert f32(pos['max'][k]) == max(p[k] for p in positions)
    idx = accessors[primitive['indices']]
    assert idx['type'] == 'SCALAR' and idx['componentType'] == 5125
    v = doc['bufferViews'][idx['bufferView']]
    assert struct.unpack_from('<III', binary, v.get('byteOffset', 0)) == (0, 1, 2)
    if 'images' in doc:
        view = doc['bufferViews'][doc['images'][0]['bufferView']]
        from io import BytesIO
        Image.open(BytesIO(binary[view['byteOffset']:view['byteOffset']+view['byteLength']])).verify()


with tempfile.TemporaryDirectory() as work:
    tmp = Path(work)
    Image.new('RGB', (4, 4), (200, 100, 50)).save(tmp / 'atlas.jpg')
    subprocess.run(['g++', '-std=c++20', '-O1', '-g', '-fsanitize=address,undefined',
                    '-fno-omit-frame-pointer', '-I', str(CPP),
                    str(Path(__file__).with_name('export_regression.cpp')),
                    str(CPP / 'export/gltf_exporter.cpp'),
                    str(CPP / 'v06/textured_glb_exporter.cpp'),
                    str(CPP / 'depth_calib.cpp'), '-o', str(tmp / 'test')], check=True)
    # This container restricts /proc; LeakSanitizer cannot enumerate threads.
    env = dict(os.environ, ASAN_OPTIONS='detect_leaks=0')
    subprocess.run([str(tmp / 'test'), str(tmp)], check=True, env=env)
    for path in sorted(tmp.glob('*.glb')):
        assert path.name != 'invalid.glb'
        inspect(path)
        print('PASS', path.name)
    assert len(list(tmp.glob('*.glb'))) == 5
print('PASS malformed mesh rejection, comma locale, unequal depth arrays; ASan/UBSan clean (leak detection disabled)')
