# -*- coding: utf-8 -*-
"""Shared configuration, feeder model, physics (IEC 60076-7 thermal model) and
standards-layer rules (IEEE C57.104-2019 DGA screening, PD and incipient-fault
rules) for the VoltyGuard full-paper study (SASG 2026).

Everything is deterministic given a seed.  Units: MW/MVAr, kV, kA, degC, ppm, pC.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data"); os.makedirs(DATA, exist_ok=True)
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
FIGS = os.path.join(ROOT, "figures"); os.makedirs(FIGS, exist_ok=True)

# ----------------------------------------------------------------------------- study design
N_DAYS = 92                     # 1 June - 31 August
H = N_DAYS * 24                 # hourly steps per realisation
N_REAL = 24                     # independent realisations (seeds 0..23)
SPLIT = {"train": list(range(0, 14)), "val": list(range(14, 18)), "test": list(range(18, 24))}
HORIZON_H = 7 * 24              # prediction target: failure within the next 7 days
DGA_ONLINE_SHARE = 0.5          # share of transformers with online DGA (daily); others monthly lab sample
PD_MONITOR_SHARE = 0.5          # share of cable joints with online PD monitoring

# ----------------------------------------------------------------------------- IEC 60076-7 (distribution ONAN)
IEC = dict(d_theta_or=55.0, d_theta_hr=23.0, R=5.0, x=0.8, y=1.6, tau_o_min=180.0, theta_ref=98.0)

def iec_top_oil_step(d_theta_o, K, dt_min=60.0, p=IEC):
    """One explicit step of the IEC 60076-7 exponential top-oil model (clause 8.2.2)."""
    ss = p["d_theta_or"] * ((1.0 + p["R"] * K ** 2) / (1.0 + p["R"])) ** p["x"]
    a = 1.0 - np.exp(-dt_min / p["tau_o_min"])
    return d_theta_o + a * (ss - d_theta_o)

def iec_hot_spot(theta_a, d_theta_o, K, p=IEC):
    return theta_a + d_theta_o + p["d_theta_hr"] * K ** p["y"]

def iec_ageing_rate(theta_h, p=IEC):
    """Relative ageing rate V for non-thermally-upgraded paper (IEC 60076-7, eq. 2)."""
    return 2.0 ** ((theta_h - p["theta_ref"]) / 6.0)

# ----------------------------------------------------------------------------- IEEE C57.104-2019 screening values (ppm)
# Table 1 (90th percentile) and Table 2 (95th percentile), O2/N2 <= 0.2, age unknown.
C57_90 = dict(H2=80, CH4=90, C2H6=90, C2H4=50, C2H2=1, CO=900)
C57_95 = dict(H2=200, CH4=150, C2H6=175, C2H4=100, C2H2=2, CO=1100)
# Table 3 rate-of-change limits (ppm/year) used for the 'rising' flag
C57_RATE = dict(H2=40, CH4=20, C2H6=30, C2H4=20, C2H2=1, CO=400)
GASES = ["H2", "CH4", "C2H6", "C2H4", "C2H2", "CO"]

def c57_status(g, rate_ppm_year):
    """DGA status 1..3 following the C57.104-2019 decision logic: 1 = all gases below the
    90th percentile, 2 = any gas above the 90th or any rate above Table 3, 3 = any gas
    above the 95th percentile with a rate above Table 3 (or any C2H2 > 95th)."""
    above90 = any(g[k] > C57_90[k] for k in GASES)
    above95 = any(g[k] > C57_95[k] for k in GASES)
    rising = any(rate_ppm_year[k] > C57_RATE[k] for k in GASES)
    if (above95 and rising) or g["C2H2"] > C57_95["C2H2"]:
        return 3
    if above90 or rising:
        return 2
    return 1

# PD and incipient-fault rules (utility practice; IEC 60270 defines the measurement, not limits)
PD_CAUTION_PC, PD_ALARM_PC = 300.0, 500.0
SCF_CAUTION_7D, SCF_ALARM_24H = 3, 3          # self-clearing (sub-cycle) fault counts
TOPOIL_ALARM_C, HOTSPOT_ALARM_C = 105.0, 140.0 # IEC 60076-7 Table 2/3 limits for distribution transformers

def standards_code(dga_status, pd_pc, scf_24h, scf_7d, topoil, hotspot):
    """Layer-1 code 0 normal / 1 caution / 2 alarm from the standards-based rules."""
    code = 0
    if dga_status == 2 or pd_pc > PD_CAUTION_PC or scf_7d >= SCF_CAUTION_7D or topoil > TOPOIL_ALARM_C - 10:
        code = 1
    if dga_status == 3 or pd_pc > PD_ALARM_PC or scf_24h >= SCF_ALARM_24H or topoil > TOPOIL_ALARM_C or hotspot > HOTSPOT_ALARM_C:
        code = 2
    return code

# ----------------------------------------------------------------------------- feeder
def build_feeder():
    """IEEE 33-bus radial feeder (Baran & Wu 1989) as shipped with pandapower, with
    realistic cable ampacities assigned from the peak-load base case."""
    net = pn.case33bw()
    pp.runpp(net, algorithm="bfsw", init="flat")
    i_peak = net.res_line.i_ka.values.copy()
    # ampacity: 1.5 x base-case current, rounded up to standard XLPE Al cable classes
    classes = np.array([0.12, 0.17, 0.23, 0.29, 0.36, 0.45])
    amp = np.array([classes[np.searchsorted(classes, min(1.5 * i, 0.449))] for i in i_peak])
    amp[~net.line.in_service.values] = 0.29  # tie lines
    net.line["max_i_ka"] = amp
    net.line["name"] = [f"L{int(a)}-{int(b)}" for a, b in zip(net.line.from_bus, net.line.to_bus)]
    return net

def feeder_edges(net):
    """Energised branches as (from, to, r, x) for the graph."""
    L = net.line[net.line.in_service]
    return list(zip(L.from_bus.astype(int), L.to_bus.astype(int), L.r_ohm_per_km * L.length_km, L.x_ohm_per_km * L.length_km, L.index))

def upstream_branch(net):
    """For each bus, the index of the energised branch feeding it (radial)."""
    up = {0: None}
    for idx, r in net.line[net.line.in_service].iterrows():
        up[int(r.to_bus)] = int(idx)
    return up

# ----------------------------------------------------------------------------- weather and load
def riyadh_summer_temperature(rng, n_days=N_DAYS):
    """Hourly ambient temperature (degC) for a Riyadh summer: daily mean ~36 degC with
    AR(1) day-to-day variation, a 16 K diurnal range (min ~03:00, max ~15:00) and 2-3
    heatwave episodes of 3-6 days adding 3-5 K.  Calibrated to June-August climate
    normals (mean maxima 43-44 degC, mean minima 28-30 degC)."""
    mu = np.zeros(n_days); dev = 0.0
    for d in range(n_days):
        dev = 0.7 * dev + rng.normal(0, 1.2); mu[d] = 36.0 + dev
    n_hw = rng.integers(2, 4)
    hw = np.zeros(n_days)
    for _ in range(n_hw):
        s = rng.integers(5, n_days - 8); L = rng.integers(3, 7); hw[s:s + L] += rng.uniform(3.0, 5.0)
    mu = mu + hw
    hours = np.arange(n_days * 24)
    diurnal = 8.0 * np.cos(2 * np.pi * (hours % 24 - 15) / 24.0)
    T = np.repeat(mu, 24) + diurnal + rng.normal(0, 0.6, len(hours))
    return T, hw

def load_profile(rng, T_amb, n_bus_loads):
    """Per-load hourly scaling factors.  Mix of residential (afternoon + evening peaks)
    and commercial (daytime) shapes, cooling-driven temperature sensitivity
    (+2.5 % per K above 30 degC), weekend reduction and lognormal noise."""
    hours = np.arange(len(T_amb)); h = hours % 24; dow = (hours // 24) % 7
    res = 0.55 + 0.30 * np.exp(-((h - 15.5) ** 2) / 8.0) + 0.25 * np.exp(-((h - 21.5) ** 2) / 6.0)
    com = 0.45 + 0.55 * np.exp(-((h - 13.0) ** 2) / 18.0)
    temp = 1.0 + 0.025 * np.clip(T_amb - 30.0, 0, None)
    wknd = np.where((dow == 4) | (dow == 5), 0.90, 1.0)       # Friday/Saturday
    out = np.zeros((n_bus_loads, len(hours)))
    mix = rng.uniform(0.5, 1.0, n_bus_loads)                    # residential share per bus
    for i in range(n_bus_loads):
        shape = mix[i] * res + (1 - mix[i]) * com
        noise = np.exp(rng.normal(0, 0.05, len(hours)))
        out[i] = 0.62 * shape * temp * wknd * noise
    return out, mix

def save_json(obj, path):
    with open(path, "w") as f: json.dump(obj, f, indent=1, default=float)
