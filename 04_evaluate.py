# -*- coding: utf-8 -*-
"""Step 04 - episode-level evaluation on the six test realisations.
Alert rule: p(fail within 7 d) >= tau for 3 consecutive hours (tau chosen on the validation
realisations to maximise episode-level F1).  Agreement gating (VoltyGuard): a CRITICAL alert
additionally requires Layer-1 code >= 1 within the previous 24 h; otherwise the alert is
routed to engineer review.  Metrics: episode recall / precision, false-alarm clusters per
100 asset-days, warning lead time per fault class, fault-vs-attack discrimination,
calibration (ECE), SERA action mapping and N-1 transfer feasibility (pandapower)."""
import json, numpy as np, pandapower as pp
from common import *

MIN_RUN = 3; LOOKBACK_D = 45; GATE_H = 24

def load_all():
    d = np.load(os.path.join(DATA, "dataset.npz"), allow_pickle=True)
    return d["X"], d["Y"], list(d["feats"]), json.loads(str(d["episodes"])), json.loads(str(d["meta"]))

def runs_of_true(mask):
    """Start indices of runs of >= MIN_RUN consecutive True (per node)."""
    H_, N_ = mask.shape; starts = {n: [] for n in range(N_)}
    for n in range(N_):
        m = mask[:, n]; c = 0
        for h in range(H_):
            c = c + 1 if m[h] else 0
            if c == MIN_RUN: starts[n].append(h - MIN_RUN + 1)
    return starts

def alert_matrix(logit, tau, l1=None):
    p = 1 / (1 + np.exp(-logit)); a = p >= tau
    if l1 is not None:                    # agating: Layer-1 code >= 1 within the last GATE_H hours
        H_ = l1.shape[0]; recent = np.zeros_like(a)
        cs = np.cumsum(np.vstack([np.zeros((1, l1.shape[1])), (l1 >= 1).astype(int)]), 0)
        idx = np.arange(H_); lo = np.maximum(0, idx - GATE_H + 1)
        recent = (cs[idx + 1] - cs[lo]) > 0; a = a & recent
    return a

def episode_metrics(logit_r, Y_r, X_r, eps, feats, tau, gate):
    """One realisation.  Returns detected list, lead times, false clusters, asset-days."""
    H_, N_ = logit_r.shape; l1 = X_r[:, :, feats.index("l1_code")] if gate else None
    a = alert_matrix(logit_r, tau, l1); starts = runs_of_true(a)
    det, miss, false_clusters = [], [], 0
    fail_eps = [e for e in eps if e["kind"] in ("joint", "transformer")]
    covered = np.zeros((H_, N_), bool)
    for e in fail_eps:
        b, f = e["bus"], e["fail"]; lo = max(0, f - LOOKBACK_D * 24)
        covered[lo:f + 24, b] = True
        s = [h for h in starts[b] if lo <= h < f]
        if s: det.append(dict(kind=e["kind"], lead_h=f - s[0], fail=f, bus=b))
        else: miss.append(dict(kind=e["kind"], fail=f, bus=b))
    for n in range(1, N_):
        for h in starts[n]:
            if not covered[h, n]: false_clusters += 1
    asset_days = (N_ - 1) * H_ / 24
    return det, miss, false_clusters, asset_days, a

def choose_tau(logits, Y, X, reals_idx, eps_all, feats, gate):
    best = (-1, 0.5)
    for tau in np.linspace(0.05, 0.95, 19):
        tp = fp = fn = 0
        for k, r in enumerate(reals_idx):
            det, miss, fc, ad, _ = episode_metrics(logits[k], Y[r], X[r], eps_all[r], feats, tau, gate)
            tp += len(det); fn += len(miss); fp += fc
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best[0]: best = (f1, float(tau))
    return best[1]

def choose_tau_budget(logits, Y, X, reals_idx, eps_all, feats, gate, budget):
    """Lowest threshold whose validation false-cluster rate stays within `budget` per 100 asset-days."""
    for tau in np.linspace(0.05, 0.95, 19):
        fc = ad = 0
        for k, r in enumerate(reals_idx):
            _, _, f_, a_, _ = episode_metrics(logits[k], Y[r], X[r], eps_all[r], feats, tau, gate); fc += f_; ad += a_
        if 100 * fc / ad <= budget: return float(tau)
    return 0.95

def discrimination(probs_r, Y_r, eps):
    """Episode-level class decision: majority of the arg-max class over the episode hours
    (excluding 'normal' votes when any anomaly vote exists)."""
    out = []
    for e in eps:
        b = e["bus"]
        if e["kind"] in ("joint", "transformer"):
            hrs = np.where(Y_r[:, b, 1] == 1)[0]; truth = 1
        elif e["kind"].startswith("fdi"):
            hrs = np.arange(e["onset"], e["end"]); truth = 2
        else:
            hrs = np.arange(e["onset"], e["end"]); truth = 3
        if len(hrs) == 0: continue
        votes = probs_r[hrs, b].argmax(-1); nz = votes[votes > 0]
        pred = int(np.bincount(nz, minlength=4).argmax()) if len(nz) else 0
        out.append(dict(kind=e["kind"], truth=truth, pred=pred, bus=b))
    return out

def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] + (1e-9 if i == bins - 1 else 0))
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)

def n1_feasible(net, bus, hour_load_scale):
    """Open the branch feeding `bus`, close the best tie switch, check voltage >= 0.95 p.u.
    and loading <= 100 % (pandapower).  Used for cable-joint failures."""
    up = upstream_branch(net); li = up[bus]
    ties = net.line.index[~net.line.in_service]
    base_ins = net.line.in_service.copy(); best = False
    net.load["scaling"] = hour_load_scale
    for t in ties:
        net.line.loc[:, "in_service"] = base_ins; net.line.at[li, "in_service"] = False; net.line.at[t, "in_service"] = True
        try:
            pp.runpp(net, algorithm="bfsw", init="flat", numba=False)
            ok = (net.res_bus.vm_pu.min() >= 0.95) and (net.res_line.loading_percent.max() <= 100) and not net.res_bus.vm_pu.isna().any()
        except Exception: ok = False
        if ok: best = True; break
    net.line.loc[:, "in_service"] = base_ins; net.load["scaling"] = 1.0
    return bool(best)


def cluster_class(P_r, h, n):
    """Majority non-normal arg-max class over the first MIN_RUN hours of an alert cluster (0 = unclassified)."""
    votes = P_r[h:h + MIN_RUN, n].argmax(-1); nz = votes[votes > 0]
    return int(np.bincount(nz, minlength=4).argmax()) if len(nz) else 0

def policy_eval(logit_r, P_r, Y_r, X_r, eps, feats, tau, gate):
    """Cyber-aware decision policy on one realisation.  Every alert cluster is routed by the class head:
    class 1 (incipient fault) or 0 (unclassified) -> maintenance scheduler; class 2 (suspected attack) -> security
    review, no switching; class 3 (sensor) -> instrumentation check.  Returns per-episode outcomes."""
    H_, N_ = logit_r.shape; l1 = X_r[:, :, feats.index("l1_code")] if gate else None
    a = alert_matrix(logit_r, tau, l1); starts = runs_of_true(a)
    route = {n: [(h, cluster_class(P_r, h, n)) for h in starts[n]] for n in range(N_)}
    out = dict(fail=[], attack=[], mask=[], false_sched=0, false_held=0)
    fail_eps = [e for e in eps if e["kind"] in ("joint", "transformer")]
    covered = np.zeros((H_, N_), bool)
    for e in fail_eps:
        b, f = e["bus"], e["fail"]; lo = max(0, f - LOOKBACK_D * 24); covered[lo:f + 24, b] = True
        al = [(h, c) for h, c in route[b] if lo <= h < f]
        sched = [h for h, c in al if c in (0, 1)]; held = [h for h, c in al if c == 2]
        status = "missed" if not al else ("scheduled" if sched and sched[0] == al[0][0] else ("delayed" if sched else "held"))
        out["fail"].append(dict(kind=e["kind"], bus=b, status=status, lead_first_h=(f - al[0][0]) if al else None,
                                lead_policy_h=(f - sched[0]) if sched else None, n_held=len(held)))
    for e in eps:
        if not e["kind"].startswith("fdi"): continue
        b = e["bus"]; al = [(h, c) for h, c in route[b] if e["onset"] <= h < e["end"]]
        covered[e["onset"]:e["end"], b] = True
        out["attack"].append(dict(kind=e["kind"], bus=b, n_alerts=len(al), n_to_scheduler=sum(1 for _, c in al if c in (0, 1)),
                                  n_held=sum(1 for _, c in al if c == 2)))
        if e["kind"] == "fdi_mask":          # simultaneous fault + attack: the masked asset is degrading towards failure
            fe = [x for x in fail_eps if x["bus"] == b and x["fail"] >= e["onset"]]
            if fe:
                f = min(x["fail"] for x in fe); lo = max(0, f - LOOKBACK_D * 24)
                sched = [h for h, c in route[b] if lo <= h < f and c in (0, 1)]
                out["mask"].append(dict(bus=b, fail_detected_and_scheduled=bool(sched), lead_policy_h=(f - sched[0]) if sched else None,
                                        flagged_as_attack=any(c == 2 for h, c in route[b] if e["onset"] <= h < e["end"])))
    for n in range(1, N_):
        for h, c in route[n]:
            if not covered[h, n]:
                if c in (0, 1): out["false_sched"] += 1
                else: out["false_held"] += 1
    return out

def tau_curve(Ls, Y, X, test, EP, feats, gate, nv):
    """Test-set recall by fault kind, precision and false-cluster rate across the threshold grid (sensitivity check)."""
    rows = []
    for tau in np.round(np.arange(0.50, 0.951, 0.05), 2):
        det, miss, fc, ad = [], [], 0, 0.0
        for k, r in enumerate(test):
            dd, mm, f_, a_, _ = episode_metrics(Ls[nv + k], Y[r], X[r], EP[r], feats, float(tau), gate); det += dd; miss += mm; fc += f_; ad += a_
        rows.append(dict(tau=float(tau), recall=len(det) / max(1, len(det) + len(miss)),
                         recall_transformer=sum(1 for d_ in det if d_["kind"] == "transformer") / max(1, sum(1 for e in det + miss if e["kind"] == "transformer")),
                         recall_joint=sum(1 for d_ in det if d_["kind"] == "joint") / max(1, sum(1 for e in det + miss if e["kind"] == "joint")),
                         precision=len(det) / max(1, len(det) + fc), false_per_100_asset_days=100 * fc / ad))
    return rows

def boot_ci(indicator_rows, B=2000, seed=0):
    """95 % percentile bootstrap of mean recall over test failure episodes (rows: seeds x episodes of 0/1), resampling episodes."""
    rng = np.random.default_rng(seed); M = np.asarray(indicator_rows, float); n = M.shape[1]
    stats = [M[:, rng.integers(0, n, n)].mean() for _ in range(B)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]

def sr_has(seed_res, k): return all(k in sr and sr[k] is not None for sr in seed_res)

def main():
    X, Y, feats, EP, META = load_all(); nv = len(SPLIT["val"]); test = SPLIT["test"]
    net = build_feeder()
    results = {}; per_model_alerts = {}
    models = {"static_alarm": ("preds_static.npz", "logit"), "static_caution": ("preds_static.npz", "logit_caution"), "xgb": ("preds_xgb.npz", "logit"),
              "gnn_full": ("preds_gnn_full.npz", "logit"), "gnn_full_gated": ("preds_gnn_full.npz", "logit"),
              "gnn_nograph": ("preds_gnn_nograph.npz", "logit"), "gnn_nophys": ("preds_gnn_nophys.npz", "logit")}
    # false-alarm budget = static alarm's false-cluster rate on the VALIDATION realisations (frozen before any test metric is read)
    d0 = np.load(os.path.join(RES, "preds_static.npz")); fc0 = ad0 = 0
    for k, r in enumerate(SPLIT["val"]):
        _, _, f_, a_, _ = episode_metrics(d0["logit"][k], Y[r], X[r], EP[r], feats, 0.5, False); fc0 += f_; ad0 += a_
    BUDGET_VAL = 100 * fc0 / ad0; results["budget_definition"] = dict(source="static_alarm validation false clusters per 100 asset-days", value=BUDGET_VAL)
    print("validation budget", round(BUDGET_VAL, 3), flush=True)
    for name, (fn, key) in models.items():
        d = np.load(os.path.join(RES, fn)); L = d[key]; P = d["probs"] if "probs" in d else None
        L = L if L.ndim == 4 else L[None]                        # [S, R', H, N]
        gate = name.endswith("_gated"); seed_res = []
        for s in range(L.shape[0]):
            Ls = L[s]
            if name.startswith("static"): tau = 0.5
            else: tau = choose_tau(Ls[:nv], Y, X, SPLIT["val"], EP, feats, gate)
            det, miss, fc, ad, disc = [], [], 0, 0.0, []
            alerts = {}
            for k, r in enumerate(test):
                dd, mm, f_, a_, amat = episode_metrics(Ls[nv + k], Y[r], X[r], EP[r], feats, tau, gate)
                det += dd; miss += mm; fc += f_; ad += a_; alerts[r] = amat
                if P is not None: disc += discrimination(P[s][nv + k] if P.ndim == 5 else P[nv + k], Y[r], EP[r])
            n_ep = len(det) + len(miss)
            lead = {k: [d_["lead_h"] / 24 for d_ in det if d_["kind"] == k] for k in ("joint", "transformer")}
            lead_all = [d_["lead_h"] / 24 for d_ in det]
            rec = {k: len(lead[k]) / max(1, sum(1 for e in det + miss if e["kind"] == k)) for k in lead}
            r_ = dict(tau=tau, episodes=n_ep, detected=len(det), recall=len(det) / max(1, n_ep), recall_by_kind=rec, false_clusters=fc,
                      precision=len(det) / max(1, len(det) + fc), false_per_100_asset_days=100 * fc / ad,
                      lead_median_d=float(np.median(lead_all)) if lead_all else None, lead_q1_d=float(np.percentile(lead_all, 25)) if lead_all else None,
                      lead_q3_d=float(np.percentile(lead_all, 75)) if lead_all else None,
                      lead_by_kind_median_d={k: (float(np.median(v)) if v else None) for k, v in lead.items()},
                      share_lead_ge_48h=float(np.mean([l >= 2 for l in lead_all])) if lead_all else None,
                      share_lead_ge_7d=float(np.mean([l >= 7 for l in lead_all])) if lead_all else None,
                      leads_d=lead_all, lead_kinds=[d_["kind"] for d_ in det])
            if disc:
                cm = np.zeros((4, 4), int)
                for d_ in disc: cm[d_["truth"], d_["pred"]] += 1
                fa = [d_ for d_ in disc if d_["truth"] in (1, 2)]
                r_["disc_confusion"] = cm.tolist(); r_["disc_accuracy_3class"] = float(np.mean([d_["truth"] == d_["pred"] for d_ in disc]))
                r_["fault_vs_attack_accuracy"] = float(np.mean([d_["truth"] == d_["pred"] for d_ in fa]))
                r_["attack_flagged_as_fault"] = int(sum(1 for d_ in disc if d_["truth"] == 2 and d_["pred"] == 1))
                r_["attack_episodes"] = int(sum(1 for d_ in disc if d_["truth"] == 2))
                r_["mask_attacks_detected"] = int(sum(1 for d_ in disc if d_["kind"] == "fdi_mask" and d_["pred"] == 2)); r_["mask_attacks"] = int(sum(1 for d_ in disc if d_["kind"] == "fdi_mask"))
            # calibration on test node-hours
            p = 1 / (1 + np.exp(-np.concatenate([Ls[nv + k][:, 1:].ravel() for k in range(len(test))])))
            y = np.concatenate([Y[r][:, 1:, 0].ravel() for r in test]); r_["ece"] = ece(p, y)
            from sklearn.metrics import roc_auc_score, average_precision_score
            r_["auroc_nodehour"] = float(roc_auc_score(y, p)); r_["auprc_nodehour"] = float(average_precision_score(y, p))
            # matched false-alarm operating point: same validation false-cluster budget as the static alarm
            if not name.startswith("static"):
                budget = BUDGET_VAL
                tau_b = choose_tau_budget(Ls[:nv], Y, X, SPLIT["val"], EP, feats, gate, budget)
                det_b, miss_b, fc_b, ad_b = [], [], 0, 0.0
                for k, r in enumerate(test):
                    dd, mm, f_, a_, _ = episode_metrics(Ls[nv + k], Y[r], X[r], EP[r], feats, tau_b, gate); det_b += dd; miss_b += mm; fc_b += f_; ad_b += a_
                lb = [d_["lead_h"] / 24 for d_ in det_b]
                r_["budget"] = dict(tau=tau_b, budget=budget, recall=len(det_b) / max(1, len(det_b) + len(miss_b)), false_per_100_asset_days=100 * fc_b / ad_b,
                                    precision=len(det_b) / max(1, len(det_b) + fc_b), lead_median_d=float(np.median(lb)) if lb else None, share_lead_ge_48h=float(np.mean([l >= 2 for l in lb])) if lb else None)
            # detection indicator per test failure episode (for the episode bootstrap)
            ind = []
            for k, r in enumerate(test):
                dd, mm, _, _, _ = episode_metrics(Ls[nv + k], Y[r], X[r], EP[r], feats, tau, gate)
                ind += [1] * len(dd) + [0] * len(mm)
            r_["detect_indicator"] = ind
            if P is not None:
                pol = dict(fail=[], attack=[], mask=[], false_sched=0, false_held=0)
                for k, r in enumerate(test):
                    o = policy_eval(Ls[nv + k], P[s][nv + k] if P.ndim == 5 else P[nv + k], Y[r], X[r], EP[r], feats, tau, gate)
                    for kk in ("fail", "attack", "mask"): pol[kk] += o[kk]
                    pol["false_sched"] += o["false_sched"]; pol["false_held"] += o["false_held"]
                st = [x["status"] for x in pol["fail"]]
                r_["policy"] = dict(n_fail=len(st), scheduled=st.count("scheduled"), delayed=st.count("delayed"), held=st.count("held"), missed=st.count("missed"),
                                    recall_policy=(st.count("scheduled") + st.count("delayed")) / max(1, len(st)),
                                    lead_policy_median_d=float(np.median([x["lead_policy_h"] / 24 for x in pol["fail"] if x["lead_policy_h"] is not None])) if any(x["lead_policy_h"] is not None for x in pol["fail"]) else None,
                                    attack_episodes=len(pol["attack"]), attack_with_alerts=sum(1 for x in pol["attack"] if x["n_alerts"] > 0),
                                    attack_sent_to_scheduler=sum(1 for x in pol["attack"] if x["n_to_scheduler"] > 0),
                                    attack_held=sum(1 for x in pol["attack"] if x["n_alerts"] > 0 and x["n_to_scheduler"] == 0),
                                    mask_cases=len(pol["mask"]), mask_fail_scheduled=sum(1 for x in pol["mask"] if x["fail_detected_and_scheduled"]),
                                    mask_flagged=sum(1 for x in pol["mask"] if x["flagged_as_attack"]),
                                    false_to_scheduler=pol["false_sched"], false_held=pol["false_held"], detail=pol)
            if name in ("gnn_full", "gnn_full_gated"): r_["tau_curve"] = tau_curve(Ls, Y, X, test, EP, feats, gate, nv)
            seed_res.append(r_); per_model_alerts[(name, s)] = alerts
        agg = {}
        for k in seed_res[0]:
            vals = [sr[k] for sr in seed_res]
            if isinstance(vals[0], (int, float)) and vals[0] is not None and not isinstance(vals[0], bool):
                agg[k] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals)) if len(vals) > 1 else 0.0)
        if "budget" in seed_res[0]:
            agg["budget"] = {k: dict(mean=float(np.mean([sr["budget"][k] for sr in seed_res if sr["budget"][k] is not None])), sd=float(np.std([sr["budget"][k] for sr in seed_res if sr["budget"][k] is not None])))
                             for k in ("recall", "false_per_100_asset_days", "precision", "lead_median_d", "share_lead_ge_48h", "tau")}
        if "policy" in seed_res[0]:
            agg["policy"] = {k: dict(mean=float(np.mean([sr["policy"][k] for sr in seed_res if sr["policy"][k] is not None])), sd=float(np.std([sr["policy"][k] for sr in seed_res if sr["policy"][k] is not None])))
                             for k in ("recall_policy", "lead_policy_median_d", "scheduled", "delayed", "held", "missed", "attack_sent_to_scheduler", "attack_held", "attack_with_alerts", "mask_fail_scheduled", "mask_flagged", "false_to_scheduler", "false_held")}
        agg["recall_ci95"] = boot_ci([sr["detect_indicator"] for sr in seed_res])
        agg["recall_by_kind"] = {k: dict(mean=float(np.mean([sr["recall_by_kind"][k] for sr in seed_res])), sd=float(np.std([sr["recall_by_kind"][k] for sr in seed_res])),
                                         per_seed=[sr["recall_by_kind"][k] for sr in seed_res]) for k in ("joint", "transformer")}
        agg["n_seeds"] = len(seed_res)
        if len(seed_res) > 3:       # like-for-like aggregate over the first three seeds (ablations were trained with three)
            agg["first3"] = {k: dict(mean=float(np.mean([sr[k] for sr in seed_res[:3]])), sd=float(np.std([sr[k] for sr in seed_res[:3]]))) for k in ("recall", "precision", "false_per_100_asset_days", "lead_median_d", "fault_vs_attack_accuracy") if sr_has(seed_res, k)}
            agg["first3"]["budget_recall"] = dict(mean=float(np.mean([sr["budget"]["recall"] for sr in seed_res[:3]])), sd=float(np.std([sr["budget"]["recall"] for sr in seed_res[:3]])))
            agg["first3"]["budget_lead"] = dict(mean=float(np.mean([sr["budget"]["lead_median_d"] for sr in seed_res[:3]])), sd=float(np.std([sr["budget"]["lead_median_d"] for sr in seed_res[:3]])))
        results[name] = dict(seeds=seed_res, agg=agg)
        print(name, {k: round(v["mean"], 3) for k, v in agg.items() if k in ("recall", "precision", "false_per_100_asset_days", "lead_median_d", "share_lead_ge_48h", "fault_vs_attack_accuracy", "ece", "auprc_nodehour")}, flush=True)
    # ---- SERA action mapping and N-1 feasibility for the gated VoltyGuard (seed 0) and the static alarm
    sera = {}
    for name in ("gnn_full_gated", "gnn_full", "static_alarm", "xgb"):
        det = results[name]["seeds"][0]; acts = dict(planned=0, transfer=0, emergency=0, missed=0)
        d = np.load(os.path.join(RES, models[name][0])); L = d[models[name][1]]; L = L if L.ndim == 4 else L[None]; Ls = L[0]; tau = det["tau"]
        n1 = []
        for k, r in enumerate(test):
            dd, mm, _, _, _ = episode_metrics(Ls[nv + k], Y[r], X[r], EP[r], feats, tau, name.endswith("_gated"))
            acts["missed"] += len(mm)
            for e in dd:
                if e["lead_h"] >= 48: acts["planned"] += 1; continue
                h = e["fail"] - e["lead_h"]
                scale = X[r][h, 1:, feats.index("p_meas")] / np.maximum(net.load.p_mw.values, 1e-6)
                if e["kind"] == "joint": feas = n1_feasible(net, e["bus"], scale)
                else:
                    i = list(net.load.bus).index(e["bus"]); S_r = X[r][h, :, feats.index("s_rated_mva")]
                    nb_bus = int(net.line.at[upstream_branch(net)[e["bus"]], "from_bus"])
                    S_load = np.hypot(X[r][h, e["bus"], feats.index("p_meas")], X[r][h, e["bus"], feats.index("q_meas")])
                    spare = S_r[nb_bus] - np.hypot(X[r][h, nb_bus, feats.index("p_meas")], X[r][h, nb_bus, feats.index("q_meas")]) if nb_bus != 0 else 0.0
                    feas = bool(spare >= S_load)
                n1.append(feas); acts["transfer" if feas else "emergency"] += 1
        sera[name] = dict(actions=acts, n1_checks=len(n1), n1_feasible=int(sum(n1)))
        if "policy" in det:
            pa = dict(planned=0, transfer=0, emergency=0, held=0, missed=0); pn1 = []
            Pm = np.load(os.path.join(RES, models[name][0]))["probs"]; Pm = Pm[0] if Pm.ndim == 5 else Pm
            for k, r in enumerate(test):
                o = policy_eval(Ls[nv + k], Pm[nv + k], Y[r], X[r], EP[r], feats, tau, name.endswith("_gated"))
                for e in o["fail"]:
                    if e["status"] == "missed": pa["missed"] += 1; continue
                    if e["status"] == "held": pa["held"] += 1; continue
                    if e["lead_policy_h"] >= 48: pa["planned"] += 1; continue
                    pa["emergency"] += 1   # short-lead cases under the policy; N-1 feasibility as above is not re-run here
            sera[name]["policy_actions"] = pa
    results["sera"] = sera
    save_json(results, os.path.join(RES, "results_eval.json")); print(json.dumps(sera, indent=1))

if __name__ == "__main__":
    main()
