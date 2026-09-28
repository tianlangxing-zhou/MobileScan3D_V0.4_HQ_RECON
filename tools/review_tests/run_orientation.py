#!/usr/bin/env python3
"""Compile and execute production Kotlin coordinate math. Requires JRE and kotlinc."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
compiler = os.environ.get('KOTLINC') or shutil.which('kotlinc')
java = os.environ.get('JAVA') or shutil.which('java')
if not compiler or not java:
    raise SystemExit('Set KOTLINC and JAVA, or put kotlinc/java on PATH.')
with tempfile.TemporaryDirectory() as work:
    jar = str(Path(work) / 'orientation.jar')
    subprocess.run([compiler, str(root/'app/src/main/java/com/mobilescan3d/ScanCoordinates.kt'),
                    str(Path(__file__).with_name('OrientationRegression.kt')),
                    '-include-runtime', '-d', jar], check=True)
    subprocess.run([java, '-jar', jar], check=True)
