# -*- coding: utf-8 -*-
"""Step 02 - assemble the learning problem from the simulated realisations.
Per node-hour: feature vector (30 raw/derived features) + rolling statistics for the
per-asset baseline; normalisation statistics from the training realisations only.
Writes data/dataset.npz with X [R,H,N,F], Y [R,H,N,3], XR [R,H,N,F+FR] (with rolling
features), the feeder adjacency, split indices, episodes and normalisation stats."""
import json, numpy as np
from common import *

ROLL_FEATS = ["topoil_resid", "hotspot_model", "topoil_meas", "C2H4", "H2", "CH4", "pd_pc", "scf_7d", "v_resid", "kcl_resid", "dga_status", "l1_code", "loading_up"]

def rolling(X, fi):
    """Mean and slope over 24 h and 168 h for selected features (causal)."""
    H_, N_, _ = X.shape; out = []
    for f in ROLL_FEATS:
        x = X[:, :, fi[f]].astype(np.float64)
        cs = np.cumsum(np.vstack([np.zeros((1, N_)), x]), 0)
        for w in (24, 168):
            idx = np.arange(H_); lo = np.maximum(0, idx - w + 1)
            mean = (cs[idx + 1] - cs[lo]) / (idx - lo + 1)[:, None]
            slope = (x - x[lo]) / np.maximum(idx - lo, 1)[:, None] * w
            out += [mean, slope]
    return np.stack(out, -1).astype(np.float32)

def main():
    net = build_feeder(); edges = feeder_edges(net)
    A = np.zeros((33, 33), np.float32)
    for a, b, r, x, _ in edges:
        w = 1.0 / np.hypot(r, x); A[a, b] = w; A[b, a] = w
    Xs, Ys, XRs, EP = [], [], [], []
    for s in range(N_REAL):
        d = np.load(os.path.join(DATA, f"real_{s:02d}.npz")); X, Y = d["X"], d["Y"]; meta = json.loads(str(d["meta"]))
        fi = {f: i for i, f in enumerate(meta["feats"])}
        Xs.append(X); Ys.append(Y); XRs.append(np.concatenate([X, rolling(X, fi)], -1)); EP.append(meta)
    X = np.stack(Xs); Y = np.stack(Ys); XR = np.stack(XRs)
    tr = SPLIT["train"]
    mu = XR[tr].reshape(-1, XR.shape[-1]).mean(0); sd = XR[tr].reshape(-1, XR.shape[-1]).std(0) + 1e-6
    feats = EP[0]["feats"] + [f"{f}_{k}{w}" for f in ROLL_FEATS for w in (24, 168) for k in ("mean", "slope")]
    np.savez_compressed(os.path.join(DATA, "dataset.npz"), X=X, Y=Y, XR=XR, A=A, mu=mu, sd=sd, feats=np.array(feats),
                        episodes=json.dumps([m["episodes"] for m in EP]), meta=json.dumps(EP, default=float))
    cls = Y[:, :, :, 1]
    stats = dict(n_real=int(N_REAL), hours=int(H), nodes=33, node_hours=int(cls.size), features_raw=len(EP[0]["feats"]), features_with_rolling=len(feats),
                 class_counts={k: int((cls == v).sum()) for k, v in {"normal": 0, "incipient": 1, "attack": 2, "sensor": 3}.items()},
                 fail7_share=float(Y[:, :, :, 0].mean()),
                 episodes={k: int(sum(1 for m in EP for e in m["episodes"] if e["kind"] == k)) for k in ["joint", "transformer", "fdi_provoke", "fdi_electrical", "fdi_mask", "sensor_stuck", "sensor_drift"]},
                 episodes_by_split={sp: {k: int(sum(1 for r in idx for e in EP[r]["episodes"] if e["kind"] == k)) for k in ["joint", "transformer", "fdi_provoke", "fdi_electrical", "fdi_mask", "sensor_stuck", "sensor_drift"]} for sp, idx in SPLIT.items()},
                 heatwave_days_mean=float(np.mean([len(m["heatwave_days"]) for m in EP])),
                 sim_runtime_s_mean=float(np.mean([m["runtime_s"] for m in EP])),
                 dga_online_share=float(np.mean([np.mean(m["dga_online"]) for m in EP])), pd_monitored_share=float(np.mean([np.mean(m["pd_monitored"]) for m in EP])))
    # static-threshold behaviour on healthy node-hours (false-alarm rate of the rules alone)
    l1 = X[:, :, 1:, EP[0]["feats"].index("l1_code")]; c = cls[:, :, 1:]
    stats["l1_rates_on_normal"] = {"caution_or_alarm": float((l1[c == 0] >= 1).mean()), "alarm": float((l1[c == 0] >= 2).mean())}
    save_json(stats, os.path.join(RES, "results_data.json")); print(json.dumps(stats, indent=1))

if __name__ == "__main__":
    main()
