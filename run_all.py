# -*- coding: utf-8 -*-
"""Runs the whole pipeline: 01 simulate -> 02 prepare -> 03 models (xgb, then networks) -> 04 evaluate -> figures -> manuscript."""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); py = sys.executable
steps = [["01_simulate.py"], ["02_prepare.py"], ["03_models.py", "--xgb"], ["03_models.py"], ["04_evaluate.py"], ["figures.py"], ["build_manuscript.py"]]
only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
for s in steps:
    if only and not s[0].startswith(only): continue
    print(">>", " ".join(s), flush=True); subprocess.run([py, os.path.join(HERE, s[0])] + s[1:], check=True, cwd=HERE)
