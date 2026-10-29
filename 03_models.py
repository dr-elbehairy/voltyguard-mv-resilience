# -*- coding: utf-8 -*-
"""Step 03 - models.
  static   : Layer-1 standards rules alone (alarm = code 2; caution = code >= 1)
  xgb      : per-asset gradient-boosted trees on the node's own features + rolling stats (no neighbours)
  gnn_full : VoltyGuard Layer 3 - GRU temporal encoder per node + 2-layer GraphSAGE message
             passing over the feeder graph, fed with raw measurements AND Layer-1/2 features
  gnn_nograph : same network with the adjacency removed (per-node only)
  gnn_nophys  : full graph network without the Layer-1/2 standards and physics features
Each network is trained with 3 seeds.  Outputs per test node-hour: p(fail within 7 d)
and class probabilities, written to results/preds_{model}.npz.
"""
import sys, time, json, numpy as np
from common import *
XGB_ONLY = "--xgb" in sys.argv
if not XGB_ONLY:
    import torch, torch.nn as nn, torch.nn.functional as Fnn
    torch.set_num_threads(8)
W = 24                       # temporal window (h)
STRIDE_TRAIN = 2
PHYS = ["hotspot_model", "topoil_model", "topoil_resid", "log_ageing_model", "v_resid", "kcl_resid", "dga_status", "l1_code"]
SEEDS = [0, 1, 2]

def load():
    d = np.load(os.path.join(DATA, "dataset.npz"), allow_pickle=True)
    return d["X"], d["Y"], d["XR"], d["A"], d["mu"], d["sd"], list(d["feats"])

class VoltyGNN((nn.Module if not XGB_ONLY else object)):
    def __init__(self, f_in, hid=48, use_graph=True, n_cls=4, drop=0.3):
        super().__init__()
        self.use_graph = use_graph
        self.gru = nn.GRU(f_in, hid, batch_first=True)
        self.lin_self1, self.lin_nb1 = nn.Linear(hid, hid), nn.Linear(hid, hid)
        self.lin_self2, self.lin_nb2 = nn.Linear(hid, hid), nn.Linear(hid, hid)
        self.drop = nn.Dropout(drop)
        self.head_fail = nn.Linear(hid, 1); self.head_cls = nn.Linear(hid, n_cls)
    def sage(self, h, An, ls, ln):
        nb = torch.matmul(An, h) if self.use_graph else torch.zeros_like(h)
        return Fnn.relu(ls(h) + ln(nb))
    def forward(self, x, An):
        # x [B, N, W, F]  An [N, N] row-normalised adjacency
        B, N_, W_, F_ = x.shape
        _, hN = self.gru(x.reshape(B * N_, W_, F_)); h = hN[-1].reshape(B, N_, -1)
        h = self.drop(self.sage(h, An, self.lin_self1, self.lin_nb1))
        h = self.drop(self.sage(h, An, self.lin_self2, self.lin_nb2))
        return self.head_fail(h).squeeze(-1), self.head_cls(h)

def windows(X, hours):
    """X [H,N,F] -> [len(hours), N, W, F] causal windows ending at each hour (zero-padded)."""
    H_, N_, F_ = X.shape; out = np.zeros((len(hours), N_, W, F_), np.float32)
    for k, h in enumerate(hours):
        lo = max(0, h - W + 1); out[k, :, W - (h - lo + 1):] = X[lo:h + 1].transpose(1, 0, 2)
    return out

def train_gnn(Xn, Y, An, use_graph, feat_idx, seed, log):
    torch.manual_seed(seed); np.random.seed(seed)
    R, H_, N_, _ = Xn.shape
    tr_hours = np.arange(W, H_, STRIDE_TRAIN)
    xs = [windows(Xn[r][:, :, feat_idx], tr_hours) for r in SPLIT["train"]]
    ys = [Y[r][tr_hours] for r in SPLIT["train"]]
    Xtr = torch.tensor(np.concatenate(xs)); Ytr = torch.tensor(np.concatenate(ys))
    pos = Ytr[:, :, 0].mean().item(); pw = torch.tensor(min(30.0, (1 - pos) / max(pos, 1e-4)))
    cls_counts = torch.bincount(Ytr[:, :, 1].long().flatten(), minlength=4).float()
    cw = (cls_counts.sum() / (4 * cls_counts.clamp(min=1))).clamp(max=20.0)
    model = VoltyGNN(len(feat_idx), use_graph=use_graph); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    A_t = torch.tensor(An)
    n = len(Xtr); bs = 64; t0 = time.time()
    xv = torch.tensor(np.concatenate([windows(Xn[r][:, :, feat_idx], np.arange(W, H_, 4)) for r in SPLIT["val"]]))
    yv = torch.tensor(np.concatenate([Y[r][np.arange(W, H_, 4)] for r in SPLIT["val"]]))
    from sklearn.metrics import average_precision_score
    best, best_state, patience = -1.0, None, 0
    for ep in range(20):
        model.train(); perm = torch.randperm(n); tot = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; xb, yb = Xtr[idx], Ytr[idx]
            xb = xb + 0.05 * torch.randn_like(xb)                      # input-noise regularisation
            lf, lc = model(xb, A_t)
            loss = Fnn.binary_cross_entropy_with_logits(lf, yb[:, :, 0], pos_weight=pw) + Fnn.cross_entropy(lc.reshape(-1, 4), yb[:, :, 1].long().reshape(-1), weight=cw)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            lf, lc = model(xv, A_t)
            vl = (Fnn.binary_cross_entropy_with_logits(lf, yv[:, :, 0], pos_weight=pw) + Fnn.cross_entropy(lc.reshape(-1, 4), yv[:, :, 1].long().reshape(-1), weight=cw)).item()
            ap = average_precision_score(yv[:, 1:, 0].numpy().ravel(), lf[:, 1:].numpy().ravel())
            acc = (lc.argmax(-1) == yv[:, :, 1].long())[yv[:, :, 1] > 0].float().mean().item()
        log.write(f"  seed {seed} ep {ep:02d} train {tot / n:.4f} val {vl:.4f} valAP {ap:.3f} anomAcc {acc:.3f} ({time.time() - t0:.0f}s)\n"); log.flush()
        score = ap + 0.2 * acc
        if score > best + 1e-4: best, best_state, patience = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 4: break
    model.load_state_dict(best_state); model.eval()
    return model, time.time() - t0, ep + 1

def predict_gnn(model, Xn, An, feat_idx, reals):
    A_t = torch.tensor(An); out_f, out_c = [], []
    with torch.no_grad():
        for r in reals:
            hours = np.arange(Xn.shape[1]); pf, pc = [], []
            for i in range(0, len(hours), 256):
                xb = torch.tensor(windows(Xn[r][:, :, feat_idx], hours[i:i + 256]))
                lf, lc = model(xb, A_t); pf.append(lf.numpy()); pc.append(torch.softmax(lc, -1).numpy())
            out_f.append(np.concatenate(pf)); out_c.append(np.concatenate(pc))
    return np.stack(out_f), np.stack(out_c)      # logits [R', H, N], probs [R', H, N, 4]

def temperature_scale(logit_val, y_val):
    """Fit a single temperature on the validation logits (Guo et al. 2017)."""
    if XGB_ONLY:
        from scipy.optimize import minimize_scalar
        lv = logit_val.reshape(-1).astype(np.float64); yv = y_val.reshape(-1).astype(np.float64)
        def nll(T):
            z = lv / T; return np.mean(np.logaddexp(0, z) - yv * z)
        return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
    lv = torch.tensor(logit_val.reshape(-1)); yv = torch.tensor(y_val.reshape(-1))
    T = torch.ones(1, requires_grad=True); opt = torch.optim.LBFGS([T], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad(); l = Fnn.binary_cross_entropy_with_logits(lv / T.clamp(min=0.05), yv); l.backward(); return l
    opt.step(closure); return float(T.clamp(min=0.05).item())

def run_xgb(XR, Y, mu, sd, log):
    import xgboost as xgb
    R, H_, N_, F_ = XR.shape
    def flat(reals, stride=1):
        xs = np.concatenate([XR[r][W::stride, 1:].reshape(-1, F_) for r in reals]); ys = np.concatenate([Y[r][W::stride, 1:].reshape(-1, 3) for r in reals]); return (xs - mu) / sd, ys
    xtr, ytr = flat(SPLIT["train"], 2); xv, yv = flat(SPLIT["val"], 4)
    t0 = time.time()
    pos = ytr[:, 0].mean(); m_f = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=min(30, (1 - pos) / pos), n_jobs=8, tree_method="hist", eval_metric="logloss", early_stopping_rounds=30)
    m_f.fit(xtr, ytr[:, 0], eval_set=[(xv, yv[:, 0])], verbose=False)
    w = np.bincount(ytr[:, 1].astype(int), minlength=4).astype(float); sw = (w.sum() / (4 * np.maximum(w, 1)))[ytr[:, 1].astype(int)]
    m_c = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=8, tree_method="hist", objective="multi:softprob", eval_metric="mlogloss", early_stopping_rounds=30)
    m_c.fit(xtr, ytr[:, 1].astype(int), sample_weight=np.minimum(sw, 20), eval_set=[(xv, yv[:, 1].astype(int))], verbose=False)
    rt = time.time() - t0; log.write(f"  xgb trained in {rt:.0f}s, best iters {m_f.best_iteration}/{m_c.best_iteration}\n")
    def pred(reals):
        lf = np.zeros((len(reals), H_, N_), np.float32); pc = np.zeros((len(reals), H_, N_, 4), np.float32); pc[:, :, :, 0] = 1
        for k, r in enumerate(reals):
            xr = ((XR[r][:, 1:].reshape(-1, F_) - mu) / sd)
            p = np.clip(m_f.predict_proba(xr)[:, 1], 1e-6, 1 - 1e-6); lf[k, :, 1:] = np.log(p / (1 - p)).reshape(H_, N_ - 1)
            pc[k, :, 1:] = m_c.predict_proba(xr).reshape(H_, N_ - 1, 4)
        return lf, pc
    return pred, rt, m_f

def main():
    X, Y, XR, A, mu, sd, feats = load()
    Xn = np.clip((XR - mu) / sd, -10, 10).astype(np.float32)
    An = A / np.maximum(A.sum(1, keepdims=True), 1e-9)       # row-normalised (mean aggregation weighted by admittance)
    raw_idx = list(range(30))                                   # the 30 hourly features; the GRU supplies the temporal context
    nophys_idx = [i for i in raw_idx if not any(feats[i].startswith(p) for p in PHYS)]
    log = open(os.path.join(RES, "train_log.txt"), "a"); summary = json.load(open(os.path.join(RES, "results_models.json"))) if os.path.exists(os.path.join(RES, "results_models.json")) else {}
    # ---- static rules
    l1 = X[:, :, :, feats.index("l1_code")]
    np.savez_compressed(os.path.join(RES, "preds_static.npz"), logit=np.where(l1 >= 2, 5.0, -5.0)[SPLIT["val"] + SPLIT["test"]].astype(np.float32),
                        logit_caution=np.where(l1 >= 1, 5.0, -5.0)[SPLIT["val"] + SPLIT["test"]].astype(np.float32), reals=np.array(SPLIT["val"] + SPLIT["test"]))
    nv = len(SPLIT["val"])
    if XGB_ONLY:
        pred, rt, m_f = run_xgb(XR, Y, mu, sd, log)
        lf, pc = pred(SPLIT["val"] + SPLIT["test"])
        T = temperature_scale(lf[:nv], Y[SPLIT["val"]][:, :, :, 0])
        np.savez_compressed(os.path.join(RES, "preds_xgb.npz"), logit=lf / T, probs=pc, reals=np.array(SPLIT["val"] + SPLIT["test"]), T=T)
        imp = m_f.feature_importances_; top = np.argsort(imp)[::-1][:12]
        summary["xgb"] = dict(train_s=rt, temperature=T, top_features=[(feats[i], float(imp[i])) for i in top])
        save_json(summary, os.path.join(RES, "results_models.json")); log.close(); print(json.dumps(summary["xgb"], indent=1)); return
    # ---- networks
    for name, use_graph, fidx in (("gnn_full", True, raw_idx), ("gnn_nograph", False, raw_idx), ("gnn_nophys", True, nophys_idx)):
        LF, PC, info = [], [], []
        for seed in SEEDS:
            log.write(f"{name} seed {seed}\n"); model, rt, n_ep = train_gnn(Xn, Y, An, use_graph, fidx, seed, log)
            lf, pc = predict_gnn(model, Xn, An, fidx, SPLIT["val"] + SPLIT["test"])
            T = temperature_scale(lf[:nv], Y[SPLIT["val"]][:, :, :, 0]); LF.append(lf / T); PC.append(pc)
            info.append(dict(seed=seed, train_s=rt, epochs=n_ep, temperature=T, n_params=sum(p.numel() for p in model.parameters())))
            if name == "gnn_full" and seed == 0: torch.save(model.state_dict(), os.path.join(RES, "gnn_full_seed0.pt"))
        np.savez_compressed(os.path.join(RES, f"preds_{name}.npz"), logit=np.stack(LF), probs=np.stack(PC), reals=np.array(SPLIT["val"] + SPLIT["test"]))
        summary[name] = info; save_json(summary, os.path.join(RES, "results_models.json"))
    log.close(); print(json.dumps(summary, indent=1, default=float))

if __name__ == "__main__":
    main()
