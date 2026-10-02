#!/usr/bin/env python3
from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CPP = ROOT / "app/src/main/cpp"
TESTS = ROOT / "tests/native_v18510"
OUT = ROOT / ".native_v18510_test_build"

compiler = shutil.which("g++") or shutil.which("clang++")
if not compiler:
    raise SystemExit("g++/clang++ not found")

eigen_candidates = [
    os.environ.get("EIGEN3_INCLUDE_DIR"),
    str(CPP / "third_party/eigen"),
]
eigen = next((Path(x) for x in eigen_candidates if x and (Path(x) / "Eigen").exists()), None)
if eigen is None:
    raise SystemExit(
        "Eigen headers not found. Keep app/src/main/cpp/third_party/eigen present "
        "or set EIGEN3_INCLUDE_DIR."
    )

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

def run(name, sources, use_eigen=False):
    exe = OUT / name
    cmd = [compiler, "-std=c++20", "-O2", "-I", str(CPP)]
    if use_eigen:
        cmd += ["-I", str(eigen)]
    cmd += [str(x) for x in sources] + ["-o", str(exe)]
    subprocess.run(cmd, check=True)
    subprocess.run([str(exe)], check=True)
    print("[PASS]", name)

run("frame_icp_test", [TESTS / "frame_icp_test.cpp"], True)
run("surfel_test", [TESTS / "surfel_test.cpp", CPP / "surfel_engine.cpp"])
run("depth_refinement_test", [TESTS / "depth_refinement_test.cpp"])
run(
    "mesh_qem_test",
    [
        TESTS / "mesh_qem_test.cpp",
        CPP / "mesh/mesh_engine.cpp",
        CPP / "tsdf_engine.cpp",
    ],
)
print("All vc18510 native regression tests passed.")
