"""
Offline pipeline smoke test — exercises the REAL modules (not a standalone
copy) against the bundled synthetic flow generator: extraction -> windowing
-> training -> rollout -> explain -> benchmark. Run this first on any fresh
checkout to confirm the environment is set up correctly.

Usage (from repo root):
    python -m tests.smoke_test
"""
import subprocess
import sys


def run(*args: str):
    print(f"\n$ {' '.join(args)}")
    result = subprocess.run(args)
    if result.returncode != 0:
        print(f"FAILED: {' '.join(args)}")
        sys.exit(1)


if __name__ == "__main__":
    run(sys.executable, "-m", "src.train", "--config", "configs/dev_synthetic.yaml")
    run(sys.executable, "-m", "evaluation.benchmark")
    run(sys.executable, "-m", "src.rollout")
    run(sys.executable, "-m", "src.explain")
    print("\nSMOKE TEST PASSED: extraction -> training -> benchmark -> rollout -> "
          "explain all ran successfully end to end against the bundled synthetic data.")
