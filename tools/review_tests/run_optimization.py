#!/usr/bin/env python3
"""Regression and optional repeatable A/B timing against an unmodified project.

python3 tools/review_tests/run_optimization.py
python3 tools/review_tests/run_optimization.py --baseline /path/to/original/MobileScan3D
Requires Python 3 and g++ (no Android/OpenCV/Ceres).
"""
from pathlib import Path
import argparse, os, subprocess, tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
sources = ['depth_calib.cpp', 'surfel_engine.cpp', 'tsdf_engine.cpp']
with tempfile.TemporaryDirectory() as work:
    def build(project, name, sanitize):
        cpp = project / 'app/src/main/cpp'
        binary = Path(work) / name
        options = ['-O1', '-g', '-fsanitize=address,undefined,float-cast-overflow',
                   '-fno-sanitize-recover=all', '-fno-omit-frame-pointer'] if sanitize else ['-O3']
        subprocess.run(['g++', '-std=c++20', *options, '-I', str(cpp),
                        str(Path(__file__).with_name('optimization_regression.cpp')),
                        *[str(cpp / s) for s in sources], '-o', str(binary)], check=True)
        return binary
    binary = build(root, 'regression', True)
    subprocess.run([str(binary)], check=True, env=dict(os.environ, ASAN_OPTIONS='detect_leaks=0'))
    if args.baseline:
        baseline = build(args.baseline.resolve(), 'baseline', False)
        optimized = build(root, 'optimized', False)
        reports = {}
        for label, path in [('baseline', baseline), ('optimized', optimized)]:
            result = subprocess.run([str(path), '--measure'], check=True, capture_output=True, text=True)
            print(label + '\n' + result.stdout, flush=True)
            reports[label] = dict(token.split('=', 1) for token in result.stdout.split() if '=' in token)
        assert reports['baseline']['fusion_digest'] == reports['optimized']['fusion_digest'], 'TSDF output changed'
        old = float(reports['baseline']['fusion_median_ms_per_frame'])
        new = float(reports['optimized']['fusion_median_ms_per_frame'])
        print(f'Host TSDF time reduction: {(1-new/old)*100:.2f}% (not an Android device measurement)')
