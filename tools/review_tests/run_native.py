#!/usr/bin/env python3
"""Run all host native regression suites in order; Android is not required."""
from pathlib import Path
import subprocess
import sys

directory = Path(__file__).resolve().parent
for name in ('run.py', 'run_geometry.py', 'run_scan_policy.py', 'run_optimization.py',
             'run_continuity.py', 'run_robustness.py', 'run_round23.py'):
    print(f'Running {name}', flush=True)
    subprocess.run([sys.executable, str(directory / name)], check=True)
print('PASS all host native suites')
