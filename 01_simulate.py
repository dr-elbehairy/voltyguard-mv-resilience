# -*- coding: utf-8 -*-
"""Step 01 - synthetic operating data for the IEEE 33-bus feeder under Riyadh summer
conditions, with (i) cable-joint degradation ending in permanent failure, (ii) thermal
overload / ageing of distribution transformers ending in failure, (iii) false-data
injection (provoke, mask/replay, stealthy electrical) and (iv) sensor malfunction.

Writes data/real_{seed:02d}.npz with
  X  [H, N, F]  node features as seen by the platform (measured + Layer-1/2 derived)
  Y  [H, N, 3]  (fail_within_7d, class 0/1/2/3, hours_to_failure or -1)
  E  episodes (json) and feature names.
Run:  python 01_simulate.py            (all seeds, multiprocessing)
      python 01_simulate.py --seed 0   (one seed)
"""
import sys, json, time
import numpy as np, pandas as pd, pandapower as pp
from multiprocessing import Pool
from common import *

FEATS = ["v_meas", "p_meas", "q_meas", "loading_up", "pin_up", "t_amb", "topoil_meas", "K_meas", "hotspot_model",
         "topoil_model", "topoil_resid", "log_ageing_model", "v_resid", "kcl_resid",
         "H2", "CH4", "C2H6", "C2H4", "C2H2", "CO", "dga_age_d", "dga_status", "pd_pc", "pd_monitored",
         "scf_24h", "scf_7d", "l1_code", "s_rated_mva", "age_years", "is_substation"]
F = len(FEATS); N = 33
CLS = {"normal": 0, "incipient": 1, "attack": 2, "sensor": 3}

def run_pf(net, p, q, r_scale, line_r0):
    net.load["p_mw"] = p; net.load["q_mvar"] = q
    net.line["r_ohm_per_km"] = line_r0 * r_scale
    pp.runpp(net, algorithm="bfsw", init="flat", numba=False)
    return net.res_bus.vm_pu.values.copy(), net.res_line.loading_percent.values.copy(), net.res_line.p_from_mw.values.copy()

def simulate(seed):
    rng = np.random.default_rng(1000 + seed)
    net = build_feeder()
    line_r0 = net.line.r_ohm_per_km.values.copy()
    load_bus = net.load.bus.values.astype(int)            # 32 load buses (1..32)
    lb = load_bus.tolist()
    p0, q0 = net.load.p_mw.values.copy(), net.load.q_mvar.values.copy()
    up = upstream_branch(net)                              # bus -> feeding line index
    children = {b: [] for b in range(N)}
    for b, li in up.items():
        if li is not None: children[int(net.line.at[li, "from_bus"])].append(b)
    T_amb, hw = riyadh_summer_temperature(rng)
    prof, mix = load_profile(rng, T_amb, len(load_bus))
    # transformer ratings: s = rated / peak apparent load
    S_peak = np.sqrt((p0 * prof.max(1)) ** 2 + (q0 * prof.max(1)) ** 2)
    sizing = rng.uniform(0.95, 1.35, len(load_bus))
    age = rng.uniform(3, 30, len(load_bus))
    # ---------------------------------------------------------------- scenario draws (node ids are load indices 0..31 -> bus load_bus[i])
    nodes = np.arange(len(load_bus))
    n_T, n_J = rng.integers(2, 4), rng.integers(2, 4)
    degr_T = rng.choice(nodes, n_T, replace=False)
    degr_J = rng.choice(np.setdiff1d(nodes, degr_T), n_J, replace=False)
    sizing[degr_T] = rng.uniform(0.68, 0.84, n_T)        # under-sized units that overload in heat waves
    S_rated = sizing * S_peak
    healthy = np.setdiff1d(nodes, np.concatenate([degr_T, degr_J]))
    episodes = []
    # joint degradation timing
    for j in degr_J:
        t0 = int(rng.uniform(5, 55) * 24); dur = int(rng.uniform(15, 38) * 24)
        episodes.append(dict(kind="joint", node=int(j), bus=int(load_bus[j]), onset=t0, fail=min(t0 + dur, H - 1), dur=dur,
                             pd_monitored=bool(rng.uniform() < PD_MONITOR_SHARE)))
    # transformer failure time chosen in the window; damage index normalised to cumulative ageing
    for t in degr_T:
        tf = int(rng.uniform(25, 88) * 24)
        episodes.append(dict(kind="transformer", node=int(t), bus=int(load_bus[t]), onset=None, fail=tf))
    # attacks
    n_A = rng.integers(2, 4); att_nodes = rng.choice(healthy, n_A, replace=False)
    for k, a in enumerate(att_nodes):
        kind = ["fdi_provoke", "fdi_electrical"][k % 2]
        t0 = int(rng.uniform(8, 80) * 24); dur = int(rng.uniform(3, 8) * 24) if kind == "fdi_provoke" else int(rng.uniform(2, 5) * 24)
        episodes.append(dict(kind=kind, node=int(a), bus=int(load_bus[a]), onset=t0, fail=None, end=min(t0 + dur, H - 1)))
    if rng.uniform() < 0.6:   # replay/mask attack on one degrading asset during its last days
        e = [e for e in episodes if e["kind"] in ("joint", "transformer")][rng.integers(0, n_T + n_J)]
        dur = int(rng.uniform(4, 9) * 24); t0 = max(e["fail"] - dur, 24 * 16)
        episodes.append(dict(kind="fdi_mask", node=e["node"], bus=e["bus"], onset=t0, fail=None, end=e["fail"], target=e["kind"]))
    # sensor malfunction
    s_nodes = rng.choice(np.setdiff1d(healthy, att_nodes), 2, replace=False)
    for k, s in enumerate(s_nodes):
        t0 = int(rng.uniform(5, 85) * 24); dur = int(rng.uniform(2, 6) * 24)
        episodes.append(dict(kind=["sensor_stuck", "sensor_drift"][k % 2], node=int(s), bus=int(load_bus[s]), onset=t0, fail=None, end=min(t0 + dur, H - 1)))
    # monitoring coverage
    dga_online = rng.uniform(size=len(load_bus)) < DGA_ONLINE_SHARE
    pd_mon = np.zeros(len(load_bus), bool)
    for e in episodes:
        if e["kind"] == "joint": pd_mon[e["node"]] = e["pd_monitored"]
    pd_mon[healthy] = rng.uniform(size=len(healthy)) < PD_MONITOR_SHARE
    lab_day = rng.integers(0, 30, len(load_bus))        # monthly lab sampling day offset
    # ---------------------------------------------------------------- state
    X = np.zeros((H, N, F), np.float32); Y = np.zeros((H, N, 3), np.float32); Y[:, :, 2] = -1
    d_oil = np.zeros(len(load_bus)); d_oil_obs = np.zeros(len(load_bus))
    gases = np.tile(np.array([20., 15., 10., 8., 0., 300.]), (len(load_bus), 1)) * rng.uniform(0.6, 1.4, (len(load_bus), 1))
    gases_meas = gases.copy(); gas_hist = {i: [(0, gases[i].copy())] for i in range(len(load_bus))}; dga_age = np.zeros(len(load_bus)); gas_rate = np.zeros((len(load_bus), 6))
    DJ = np.zeros(len(load_bus)); DT = np.zeros(len(load_bus)); cum_age = np.zeros(len(load_bus))
    scf_log = np.zeros((H, len(load_bus)))               # recorded self-clearing faults per node-hour
    r_scale = np.ones(len(net.line))
    failed_T = {}; failed_J = {}
    # pre-compute transformer cumulative ageing to normalise DT (depends on load & ambient only)
    pre_oil = np.zeros(len(load_bus)); pre_cum = np.zeros((H, len(load_bus)))
    for h in range(H):
        S = np.sqrt((p0 * prof[:, h]) ** 2 + (q0 * prof[:, h]) ** 2); K = S / S_rated
        pre_oil = iec_top_oil_step(pre_oil, K); th = iec_hot_spot(T_amb[h], pre_oil, K)
        pre_cum[h] = (pre_cum[h - 1] if h else 0) + iec_ageing_rate(th)
    A_fail = {e["node"]: pre_cum[e["fail"], e["node"]] for e in episodes if e["kind"] == "transformer"}
    ep_T = {e["node"]: e for e in episodes if e["kind"] == "transformer"}
    ep_J = {e["node"]: e for e in episodes if e["kind"] == "joint"}
    def active(kind_set, node, h):
        for e in episodes:
            if e["kind"] in kind_set and e["node"] == node and e["onset"] is not None and e["onset"] <= h < (e["end"] if e.get("end") else H):
                return e
        return None
    replay_buf = {}
    t_start = time.time()
    for h in range(H):
        # ---- true physical state
        p = p0 * prof[:, h]; q = q0 * prof[:, h]
        # joint damage growth (accelerating, heating-dependent)
        for j, e in ep_J.items():
            if j in failed_J: continue
            if h >= e["onset"]:
                tau = (h - e["onset"]) / e["dur"]
                DJ[j] = min(1.05, DJ[j] + (1.5 * max(tau, 1e-3) ** 0.5 / e["dur"]) * (0.7 + 0.6 * (prof[j, h] / prof[j].max()) ** 2))
                r_scale[up[load_bus[j]]] = 1.0 + 0.6 * DJ[j] ** 2
        v_true, load_true, pin_true = run_pf(net, p, q, r_scale, line_r0)
        S = np.sqrt(p ** 2 + q ** 2); K = S / S_rated
        d_oil = iec_top_oil_step(d_oil, K); theta_o = T_amb[h] + d_oil; theta_h = iec_hot_spot(T_amb[h], d_oil, K)
        V = iec_ageing_rate(theta_h); cum_age += V
        for t, e in ep_T.items():
            if t not in failed_T: DT[t] = min(1.05, cum_age[t] / A_fail[t])
        # gas generation (ppm/h): thermal base + defect contribution
        base = 0.0006 * np.exp(0.035 * (theta_h - 98.0))
        defect = np.where(DT > 0.5, 1.6 * (DT - 0.5) ** 2 * 30.0 / 24.0, 0.0)
        inc = np.zeros_like(gases)
        inc[:, 3] = base + defect                                   # C2H4
        inc[:, 1] = 0.6 * (base + defect); inc[:, 2] = 0.4 * (base + defect)
        inc[:, 0] = 0.15 * base + 1.2 * defect                      # H2
        inc[:, 4] = np.where(DT > 0.85, 0.05 * defect, 0.0)         # C2H2 (T3)
        inc[:, 5] = 0.3 * base                                      # CO
        gases += inc * np.exp(rng.normal(0, 0.1, gases.shape))
        # self-clearing faults (true events) at degrading joints and rare background events
        lam = (0.005 + 4.0 * DJ ** 3) / 24.0
        ev = rng.poisson(lam)
        for j in np.where(ev > 0)[0]:
            tgt = j
            if rng.uniform() > 0.8:                                  # localisation error: neighbouring segment
                nb = [c for c in children[load_bus[j]]] + [int(net.line.at[up[load_bus[j]], "from_bus"])]
                nb = [lb.index(b) for b in nb if b in lb]
                if nb: tgt = int(rng.choice(nb))
            scf_log[h, tgt] += ev[j]
        # ---- failures
        for j, e in ep_J.items():
            if j not in failed_J and (DJ[j] >= 1.0 or h >= e["fail"]):
                failed_J[j] = h; e["fail"] = h; DJ[j] = 0.0; r_scale[up[load_bus[j]]] = 1.0
        for t, e in ep_T.items():
            if t not in failed_T and (DT[t] >= 1.0 or h >= e["fail"]):
                failed_T[t] = h; e["fail"] = h; DT[t] = 0.0; cum_age[t] = 0; A_fail[t] = np.inf
                gases[t] = np.array([20., 15., 10., 8., 0., 300.])
        # ---- measurements (with noise)
        v_meas = v_true * (1 + rng.normal(0, 0.003, N))
        p_meas = p * (1 + rng.normal(0, 0.01, len(p))); q_meas = q * (1 + rng.normal(0, 0.01, len(q)))
        pin_meas = pin_true * (1 + rng.normal(0, 0.01, len(pin_true))); load_meas = load_true * (1 + rng.normal(0, 0.01, len(load_true)))
        topoil_meas = theta_o + rng.normal(0, 1.0, len(theta_o))
        # DGA sampling: online daily (00:00), lab monthly
        if h % 24 == 0:
            day = h // 24
            samp = dga_online | ((day % 30) == lab_day)
            new = gases * np.exp(rng.normal(0, 0.05, gases.shape))
            gases_meas[samp] = new[samp]; dga_age = np.where(samp, 0, dga_age + 1)
            for i in np.where(samp)[0]:
                gas_hist[i].append((day, new[i].copy()))
                gas_hist[i] = [(dd, g) for (dd, g) in gas_hist[i] if day - dd <= 60]
                dd = np.array([d_ for (d_, g) in gas_hist[i]], float)
                if len(dd) >= 3 and dd.max() - dd.min() >= 21:        # trend over >= 3 samples spanning >= 3 weeks
                    G = np.array([g for (d_, g) in gas_hist[i]])
                    slope = np.polyfit(dd, G, 1)[0]                      # ppm/day per gas
                    gas_rate[i] = slope * 365.0
        pd_pc = np.where(pd_mon, 40.0 + 700.0 * DJ ** 2 + rng.normal(0, 15, len(DJ)), 0.0)
        # ---- attacks and sensor faults applied to the measured channels
        cls = np.zeros(N, int)
        for i in range(len(load_bus)):
            b = load_bus[i]
            e = active({"fdi_provoke", "fdi_electrical", "fdi_mask", "sensor_stuck", "sensor_drift"}, i, h)
            if e is None: continue
            if e["kind"] == "fdi_provoke":
                if "dT" not in e: e["dT"] = float(rng.uniform(20, 35)); e["gf"] = float(rng.uniform(3, 6))
                frac = (h - e["onset"]) / max(1, e["end"] - e["onset"])
                topoil_meas[i] += e["dT"] * min(1.0, 0.3 + frac)
                if dga_online[i] and h % 24 == 0: gases_meas[i] *= 1.0 + (e["gf"] - 1.0) * frac
                cls[b] = CLS["attack"]
            elif e["kind"] == "fdi_electrical":
                if "f" not in e: e["f"] = float(rng.uniform(1.4, 1.8))
                dp = p_meas[i] * (e["f"] - 1); dq = q_meas[i] * (e["f"] - 1)
                p_meas[i] *= e["f"]; q_meas[i] *= e["f"]
                li = up[b]; z = complex(net.line.at[li, "r_ohm_per_km"], net.line.at[li, "x_ohm_per_km"])
                dv = (dp * z.real + dq * z.imag) / (12.66 ** 2)      # consistent local voltage drop (p.u.)
                v_meas[b] -= dv; pin_meas[li] += dp; load_meas[li] *= e["f"]
                cls[b] = CLS["attack"]
            elif e["kind"] == "fdi_mask":
                key = (i, h - 15 * 24)
                if key in replay_buf:
                    topoil_meas[i], gases_meas[i], pd_pc[i] = replay_buf[key]
                    scf_log[h, i] = 0
                cls[b] = CLS["attack"]
            elif e["kind"] == "sensor_stuck":
                if "val" not in e: e["val"] = float(topoil_meas[i])
                topoil_meas[i] = e["val"]; cls[b] = CLS["sensor"]
            elif e["kind"] == "sensor_drift":
                topoil_meas[i] += 0.4 * (h - e["onset"]); cls[b] = CLS["sensor"]
        for i in range(len(load_bus)):
            replay_buf[(i, h)] = (float(topoil_meas[i]), gases_meas[i].copy(), float(pd_pc[i]))
            replay_buf.pop((i, h - 16 * 24), None)
        # ---- Layer 2: physics observers computed from MEASURED data
        S_m = np.sqrt(p_meas ** 2 + q_meas ** 2); K_m = S_m / S_rated
        d_oil_obs = iec_top_oil_step(d_oil_obs, K_m); topoil_model = T_amb[h] + d_oil_obs; hot_model = iec_hot_spot(T_amb[h], d_oil_obs, K_m)
        v_pf, _, pin_pf = run_pf(net, p_meas, q_meas, np.ones(len(net.line)), line_r0)
        v_resid = v_meas - v_pf
        # power-balance residual: measured branch flow minus the flow implied by all measured loads (kW)
        kcl = np.zeros(N)
        for b in range(1, N):
            kcl[b] = pin_meas[up[b]] - pin_pf[up[b]]
        # ---- Layer 1 standards codes
        scf24 = scf_log[max(0, h - 23):h + 1].sum(0); scf7 = scf_log[max(0, h - 167):h + 1].sum(0)
        for i in range(len(load_bus)):
            b = load_bus[i]
            g = dict(zip(GASES, gases_meas[i])); rt = dict(zip(GASES, gas_rate[i]))
            st = c57_status(g, rt)
            l1 = standards_code(st, pd_pc[i], scf24[i], scf7[i], topoil_meas[i], hot_model[i])
            X[h, b] = [v_meas[b], p_meas[i], q_meas[i], load_meas[up[b]], pin_meas[up[b]], T_amb[h], topoil_meas[i], K_m[i], hot_model[i],
                       topoil_model[i], topoil_meas[i] - topoil_model[i], np.log10(iec_ageing_rate(hot_model[i])), v_resid[b] * 100, kcl[b] * 1000,
                       *gases_meas[i], dga_age[i], st, pd_pc[i], float(pd_mon[i]), scf24[i], scf7[i], l1, S_rated[i], age[i], 0.0]
            # labels
            c = cls[b]
            ttf = -1
            if i in ep_J and i not in failed_J and h >= ep_J[i]["onset"]:
                ttf = ep_J[i]["fail"] - h
                if c == 0 and DJ[i] >= 0.35: c = CLS["incipient"]
            if i in ep_T and i not in failed_T:
                ttf = ep_T[i]["fail"] - h
                if c == 0 and DT[i] >= 0.35: c = CLS["incipient"]
            Y[h, b] = [float(0 <= ttf <= HORIZON_H), c, ttf]
        X[h, 0] = [v_meas[0], 0, 0, load_meas[0], pin_meas[0], T_amb[h], 0, 0, 0, 0, 0, 0, v_resid[0] * 100, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1.0]
    meta = dict(seed=seed, episodes=episodes, heatwave_days=[int(d) for d in np.where(hw > 0)[0]], sizing=sizing.tolist(),
                dga_online=dga_online.tolist(), pd_monitored=pd_mon.tolist(), runtime_s=time.time() - t_start, feats=FEATS)
    np.savez_compressed(os.path.join(DATA, f"real_{seed:02d}.npz"), X=X, Y=Y, meta=json.dumps(meta, default=float))
    return seed, meta["runtime_s"], [(e["kind"], e["node"], e.get("onset"), e.get("fail"), e.get("end")) for e in episodes]

if __name__ == "__main__":
    if "--seed" in sys.argv:
        s = int(sys.argv[sys.argv.index("--seed") + 1]); print(simulate(s))
    else:
        with Pool(6) as pool:
            for r in pool.imap_unordered(simulate, range(N_REAL)): print(r, flush=True)
