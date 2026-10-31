# -*- coding: utf-8 -*-
"""Figures for the full paper (300 dpi PNG, grey-scale friendly).
Fig. 2  example realisation: precursors of a transformer and a cable-joint failure with the
        static rule, the per-asset model and VoltyGuard alerts
Fig. 3  warning lead-time distributions per model and fault class
Fig. 4  fault-vs-attack discrimination (confusion) and reliability diagram
Fig. 5  ablation: recall / false alarms / lead time for the graph and physics layers"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
C = {"static": "#7f7f7f", "xgb": "#1f77b4", "gnn": "#b22222", "gated": "#2ca02c"}
R = json.load(open(os.path.join(RES, "results_eval.json")))
d = np.load(os.path.join(DATA, "dataset.npz"), allow_pickle=True); X, Y, feats = d["X"], d["Y"], list(d["feats"]); EP = json.loads(str(d["episodes"]))
fi = {f: i for i, f in enumerate(feats)}; nv = len(SPLIT["val"]); test = SPLIT["test"]

def sig(x): return 1 / (1 + np.exp(-x))

def fig2():
    Lg = np.load(os.path.join(RES, "preds_gnn_full.npz"))["logit"][0]; Lx = np.load(os.path.join(RES, "preds_xgb.npz"))["logit"]
    tau_g = R["gnn_full"]["seeds"][0]["tau"]; tau_x = R["xgb"]["seeds"][0]["tau"]
    # pick one transformer and one joint episode in the test set with the longest observable history
    cands = [(k, r, e) for k, r in enumerate(test) for e in EP[r] if e["kind"] in ("transformer", "joint") and e["fail"] > 30 * 24]
    et = [c for c in cands if c[2]["kind"] == "transformer"][0]; ej = [c for c in cands if c[2]["kind"] == "joint" and c[2].get("pd_monitored")][0]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2))
    for col, (k, r, e) in enumerate((et, ej)):
        b, f = e["bus"], e["fail"]; lo = max(0, f - 40 * 24); t = (np.arange(lo, f + 12) - f) / 24
        x = X[r][lo:f + 12, b]
        ax = axes[0, col]
        if e["kind"] == "transformer":
            ax.plot(t, x[:, fi["hotspot_model"]], color="k", lw=0.8, label="Hot-spot (IEC 60076-7 model)")
            ax.plot(t, x[:, fi["topoil_meas"]], color="#555", lw=0.6, ls="--", label="Top-oil (measured)")
            ax.axhline(HOTSPOT_ALARM_C, color="k", lw=0.5, ls=":"); ax.set_ylabel("Temperature (°C)")
            ax2 = ax.twinx(); ax2.plot(t, x[:, fi["C2H4"]], color=C["gnn"], lw=0.9, label="C$_2$H$_4$ (ppm)"); ax2.axhline(C57_95["C2H4"], color=C["gnn"], lw=0.5, ls=":")
            ax2.set_ylabel("C$_2$H$_4$ (ppm)", color=C["gnn"]); ax2.spines["top"].set_visible(False)
            ax.set_title(f"(a) Transformer at bus {b}: thermal ageing", loc="left")
        else:
            ax.plot(t, x[:, fi["pd_pc"]], color="k", lw=0.8, label="Partial discharge (pC)"); ax.axhline(PD_ALARM_PC, color="k", lw=0.5, ls=":"); ax.set_ylabel("PD (pC)")
            ax2 = ax.twinx(); ax2.step(t, x[:, fi["scf_7d"]], color=C["gnn"], lw=0.9, label="Self-clearing faults / 7 d"); ax2.set_ylabel("Events / 7 d", color=C["gnn"]); ax2.spines["top"].set_visible(False)
            ax.set_title(f"(b) Cable joint at bus {b}: incipient faults", loc="left")
        h1, l1_ = ax.get_legend_handles_labels(); h2, l2_ = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1_ + l2_, loc="upper left", frameon=False)
        ax = axes[1, col]
        ax.plot(t, sig(Lg[nv + k][lo:f + 12, b]), color=C["gnn"], lw=0.9, label="VoltyGuard p(failure ≤ 7 d)")
        ax.plot(t, sig(Lx[nv + k][lo:f + 12, b]), color=C["xgb"], lw=0.8, label="Per-asset GBT")
        ax.step(t, x[:, fi["l1_code"]] / 2, color=C["static"], lw=0.8, where="post", label="Standards code / 2")
        ax.axhline(tau_g, color=C["gnn"], lw=0.5, ls=":"); ax.axvline(0, color="k", lw=0.6); ax.axvline(-2, color="k", lw=0.5, ls="--")
        ax.text(-2.1, 1.02, "48 h", ha="right", va="bottom", fontsize=6.5)
        ax.set_ylim(0, 1.08); ax.set_xlabel("Days to failure"); ax.set_ylabel("Probability / code")
        if col == 0: ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig2_precursors.png")); plt.close(fig)

def fig3():
    names = [("static_alarm", "Static alarm"), ("static_caution", "Static caution"), ("xgb", "Per-asset GBT"), ("gnn_full", "VoltyGuard"), ("gnn_full_gated", "VoltyGuard (gated)")]
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    data, labels, cols = [], [], []
    for n, lab in names:
        s = R[n]["seeds"][0]; data.append(s["leads_d"] if s["leads_d"] else [0]); labels.append(f"{lab}\n(n={s['detected']}/{s['episodes']})")
        cols.append(C["static"] if "static" in n else C["xgb"] if n == "xgb" else C["gated"] if "gated" in n else C["gnn"])
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.55, showfliers=True, flierprops=dict(marker=".", ms=3))
    for p, c in zip(bp["boxes"], cols): p.set_facecolor(c); p.set_alpha(0.35); p.set_edgecolor(c)
    for m in bp["medians"]: m.set_color("k")
    ax.axhline(2, color="k", lw=0.6, ls="--"); ax.text(0.55, 2.2, "SERA 2-day notice", fontsize=6.5)
    ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(labels, fontsize=6); ax.set_ylabel("Warning lead time (days)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig3_leadtime.png")); plt.close(fig)

def fig4():
    s = R["gnn_full"]["seeds"][0]; cm = np.array(s["disc_confusion"])[1:, :]   # rows truth 1..3, cols pred 0..3
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 5.2))
    ax = axes[0]; im = ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Greys", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="w" if cm[i, j] / cm[i].sum() > 0.5 else "k", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(["normal", "incipient\nfault", "data\nattack", "sensor\nfault"]); ax.set_yticks(range(3)); ax.set_yticklabels(["incipient fault", "data attack", "sensor fault"])
    ax.set_xlabel("Predicted (episode majority)"); ax.set_ylabel("True episode"); ax.set_title("(a) Anomaly discrimination, test episodes", loc="left")
    # reliability diagram
    ax = axes[1]
    Lg = np.load(os.path.join(RES, "preds_gnn_full.npz"))["logit"][0]; Lx = np.load(os.path.join(RES, "preds_xgb.npz"))["logit"]
    for L, lab, c in ((Lg, "VoltyGuard", C["gnn"]), (Lx, "Per-asset GBT", C["xgb"])):
        p = sig(np.concatenate([L[nv + k][:, 1:].ravel() for k in range(len(test))])); y = np.concatenate([Y[r][:, 1:, 0].ravel() for r in test])
        edges = np.linspace(0, 1, 11); xs, ys, ns = [], [], []
        for i in range(10):
            m = (p >= edges[i]) & (p < edges[i + 1] + (1e-9 if i == 9 else 0))
            if m.sum() > 20: xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
        ax.plot(xs, ys, "o-", color=c, ms=3, lw=0.9, label=f"{lab} (ECE {R['gnn_full' if lab=='VoltyGuard' else 'xgb']['seeds'][0]['ece']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.6); ax.set_xlabel("Predicted probability of failure within 7 d"); ax.set_ylabel("Observed frequency"); ax.legend(frameon=False, loc="upper left")
    ax.set_title("(b) Reliability after temperature scaling", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig4_discrimination_calibration.png")); plt.close(fig)

def fig5():
    names = [("gnn_full", "Full"), ("gnn_nograph", "No graph"), ("gnn_nophys", "No standards/\nphysics layers"), ("xgb", "Per-asset\nGBT"), ("static_alarm", "Static\nalarm")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for ax, key, lab in zip(axes, ("recall", "false_per_100_asset_days", "lead_median_d"), ("Episode recall", "False alarms per 100 asset-days", "Median lead time (days)")):
        def _g(n, stat):   # like-for-like: first three seeds when a model has more
            a = R[n]["agg"]
            if "first3" in a and key in a["first3"]: return a["first3"][key][stat]
            return a[key][stat] if key in a else 0
        m = [_g(n, "mean") for n, _ in names]; s = [_g(n, "sd") for n, _ in names]
        ax.bar(range(len(names)), m, yerr=s, color=[C["gnn"], "#e08080", "#e0a0a0", C["xgb"], C["static"]], capsize=2, width=0.65)
        ax.set_xticks(range(len(names))); ax.set_xticklabels([l for _, l in names], fontsize=6); ax.set_title(lab, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig5_ablation.png")); plt.close(fig)

if __name__ == "__main__":
    fig2(); fig3(); fig4(); fig5(); print("figures written")
