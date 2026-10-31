# -*- coding: utf-8 -*-
"""Step 03b - two additional training seeds (3, 4) for the full VoltyGuard network only,
appended to preds_gnn_full.npz and results_models.json (added in response to the
co-author review of 31 Oct 2026: seed-specific failure at the F1 operating point)."""
import importlib.util, sys, os, json, numpy as np, torch
spec = importlib.util.spec_from_file_location("m03", os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_models.py"))
m03 = importlib.util.module_from_spec(spec); sys.modules["m03"] = m03; spec.loader.exec_module(m03)
from common import *
EXTRA = [3, 4]
X, Y, XR, A, mu, sd, feats = m03.load()
Xn = np.clip((XR - mu) / sd, -10, 10).astype(np.float32); An = A / np.maximum(A.sum(1, keepdims=True), 1e-9)
raw_idx = list(range(30)); nv = len(SPLIT["val"])
log = open(os.path.join(RES, "train_log.txt"), "a"); summary = json.load(open(os.path.join(RES, "results_models.json")))
old = np.load(os.path.join(RES, "preds_gnn_full.npz")); LF = list(old["logit"]); PC = list(old["probs"]); info = list(summary["gnn_full"])
done = {i["seed"] for i in info}
for seed in EXTRA:
    if seed in done: continue
    log.write(f"gnn_full seed {seed} (extra)\n"); model, rt, n_ep = m03.train_gnn(Xn, Y, An, True, raw_idx, seed, log)
    lf, pc = m03.predict_gnn(model, Xn, An, raw_idx, SPLIT["val"] + SPLIT["test"])
    T = m03.temperature_scale(lf[:nv], Y[SPLIT["val"]][:, :, :, 0]); LF.append(lf / T); PC.append(pc)
    info.append(dict(seed=seed, train_s=rt, epochs=n_ep, temperature=T, n_params=sum(p.numel() for p in model.parameters())))
    print("seed", seed, "done", round(rt), "s", flush=True)
np.savez_compressed(os.path.join(RES, "preds_gnn_full.npz"), logit=np.stack(LF), probs=np.stack(PC), reals=old["reals"])
summary["gnn_full"] = info; save_json(summary, os.path.join(RES, "results_models.json")); log.close(); print("saved", len(LF), "seeds")
