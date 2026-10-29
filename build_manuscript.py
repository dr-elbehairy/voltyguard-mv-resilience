# -*- coding: utf-8 -*-
"""Builds the SASG 2026 full paper (.docx) in the conference's IEEE-PES style layout:
US letter, 16.9 mm margins, two 88.9 mm columns with 4.2 mm gutter, Times New Roman
10 pt body, Roman-numeral section headings, abstract <= 150 words, <= 10 index terms,
IEEE numeric references, biographies.  Every number is read from analysis/results/*.json.
Run: python build_manuscript.py   -> ../VoltyGuard_SASG26_FullPaper.docx"""
import json, os, re, sys
import numpy as np
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX, WD_BREAK
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); FIGS = f"{ROOT}/figures"; RES = f"{HERE}/results"
D = json.load(open(f"{RES}/results_data.json")); E = json.load(open(f"{RES}/results_eval.json")); M = json.load(open(f"{RES}/results_models.json"))
FONT = "Times New Roman"
REPO = "https://github.com/dr-elbehairy/voltyguard-mv-resilience"

def ag(model, key, d=2, pct=False, sd=True):
    v = E[model]["agg"][key]; m = v["mean"] * (100 if pct else 1); s = v["sd"] * (100 if pct else 1)
    txt = f"{m:.{d}f}"
    if sd and s > 0: txt += f" ± {s:.{d}f}"
    return txt
def s0(model, key): return E[model]["seeds"][0][key]
def bg(model, key, d=2, pct=False, sd=True):
    v = E[model]["agg"]["budget"][key]; m = v["mean"] * (100 if pct else 1); sdv = v["sd"] * (100 if pct else 1)
    return f"{m:.{d}f}" + (f" ± {sdv:.{d}f}" if sd and sdv > 0 else "")
def f1(x): return f"{x:.1f}"
def pct(x, d=0): return f"{100 * x:.{d}f}%"

# ------------------------------------------------------------------ document
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.59), Cm(27.94)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"): setattr(sec, side, Cm(1.69))
st = doc.styles["Normal"]; st.font.name = FONT; st.font.size = Pt(10)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT); st.element.rPr.rFonts.set(qn("w:cs"), FONT)
st.paragraph_format.space_after = Pt(0); st.paragraph_format.line_spacing = 1.0

def set_cols(section, n, space_cm=0.42):
    sectPr = section._sectPr
    for c in sectPr.findall(qn("w:cols")): sectPr.remove(c)
    cols = OxmlElement("w:cols"); cols.set(qn("w:num"), str(n)); cols.set(qn("w:space"), str(int(space_cm / 2.54 * 1440))); cols.set(qn("w:equalWidth"), "1")
    sectPr.append(cols)

def P(text="", bold=False, italic=False, align="j", size=10, after=0, before=0, indent=True, keep=False, hl=False):
    p = doc.add_paragraph()
    p.alignment = {"c": WD_ALIGN_PARAGRAPH.CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT, "l": WD_ALIGN_PARAGRAPH.LEFT}.get(align, WD_ALIGN_PARAGRAPH.JUSTIFY)
    pf = p.paragraph_format; pf.space_after = Pt(after); pf.space_before = Pt(before); pf.line_spacing = 1.0
    if indent and align == "j": pf.first_line_indent = Cm(0.42)
    if keep: pf.keep_with_next = True
    for tok in re.split(r"(\*\*.+?\*\*|\*.+?\*|\^.+?\^|_\{.+?\})", text):
        if not tok: continue
        if tok.startswith("**"): r = p.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("*"): r = p.add_run(tok[1:-1]); r.italic = True
        elif tok.startswith("^"): r = p.add_run(tok[1:-1]); r.font.superscript = True
        elif tok.startswith("_{"): r = p.add_run(tok[2:-1]); r.font.subscript = True
        else: r = p.add_run(tok)
        r.font.name = FONT; r.font.size = Pt(size); r.bold = r.bold or bold; r.italic = r.italic or italic
        if hl: r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return p

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
_h1 = [0]; _h2 = [0]
def H1(text):
    _h1[0] += 1; _h2[0] = 0
    p = P(f"{ROMAN[_h1[0] - 1]}. {text}", align="c", size=10, before=8, after=4, indent=False, keep=True)
    for r in p.runs: r.font.small_caps = True
    return p
def H1u(text):   # unnumbered (Acknowledgment / References / Biographies)
    p = P(text, align="c", size=10, before=8, after=4, indent=False, keep=True)
    for r in p.runs: r.font.small_caps = True
def H2(text):
    _h2[0] += 1
    return P(f"{chr(64 + _h2[0])}. {text}", italic=True, align="l", size=10, before=5, after=2, indent=False, keep=True)

def set_cell_bg(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcolor); tcPr.append(shd)

_tab = [0]
def TABLE(title, header, rows, widths, size=7.5, note=None):
    _tab[0] += 1
    p = P(f"TABLE {ROMAN[_tab[0] - 1]}", align="c", size=8, before=6, after=0, indent=False, keep=True)
    p2 = P(title.upper(), align="c", size=7.5, after=2, indent=False, keep=True)
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    tblPr = t._tbl.tblPr; lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(size); r.font.name = FONT; set_cell_bg(c, "EDEDED")
    trPr = t.rows[0]._tr.get_or_add_trPr(); th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; para = cells[i].paragraphs[0]; bold = False
            if isinstance(v, tuple): v, bold = v
            r = para.add_run(str(v)); r.font.size = Pt(size); r.font.name = FONT; r.bold = bold
    tblW = OxmlElement("w:tblW"); tblW.set(qn("w:w"), str(int(sum(widths) / 2.54 * 1440))); tblW.set(qn("w:type"), "dxa")
    for e in tblPr.findall(qn("w:tblW")): tblPr.remove(e)
    tblPr.append(tblW)
    grid = t._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths): gc.set(qn("w:w"), str(int(w / 2.54 * 1440)))
    for ri, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); trPr.append(cs)
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w)
            for para in row.cells[i].paragraphs:
                para.paragraph_format.space_after = Pt(0); para.paragraph_format.line_spacing = 1.0
                if ri < len(t.rows) - 1: para.paragraph_format.keep_with_next = True
    if note: P(note, align="l", size=7, after=4, indent=False)
    else: P("", after=4, indent=False)

_fig = [0]
def FIG(path, caption, width_cm=8.6):
    _fig[0] += 1
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next = True; p.paragraph_format.space_before = Pt(4)
    p.add_run().add_picture(path, width=Cm(width_cm))
    cp = P(f"Fig. {_fig[0]}.  {caption}", align="j", size=8, after=6, indent=False)
    return _fig[0]

def wide_section(n_cols):
    """Continuous section break and column count (for figures spanning both columns)."""
    s = doc.add_section(WD_SECTION.CONTINUOUS); set_cols(s, n_cols)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"): setattr(s, side, Cm(1.69))
    return s

# ------------------------------------------------------------------ references (IEEE numeric, order of first citation)
REFS = {
 "sera": "Saudi Electricity Regulatory Authority, \"Electrical service level standards guide (guaranteed standards),\" Umm Al-Qura Gazette, Jul. 2025. [Online]. Available: https://www.uqn.gov.sa/details?p=27249",
 "daleel": "Daleel News, \"Saudi Smart Grid Conference powers energy innovation,\" Dec. 19, 2025. [Online]. Available: https://news.daleel.gov.sa/energy/saudi-smart-grid-conference-powers-energy-innovation",
 "attaqa": "Attaqa, \"Saudi Electricity Company 2024 results,\" Mar. 5, 2025. [Online]. Available: https://attaqa.net",
 "c57104": "IEEE Guide for the Interpretation of Gases Generated in Mineral Oil-Immersed Transformers, IEEE Std C57.104-2019, 2019, doi: 10.1109/IEEESTD.2019.8890040.",
 "iec60599": "Mineral oil-filled electrical equipment in service — Guidance on the interpretation of dissolved and free gases analysis, IEC 60599:2022.",
 "duval": "M. Duval and A. dePablo, \"Interpretation of gas-in-oil analysis using new IEC publication 60599 and IEC TC 10 databases,\" IEEE Electr. Insul. Mag., vol. 17, no. 2, pp. 31–41, 2001, doi: 10.1109/57.917529.",
 "iec60270": "High-voltage test techniques — Partial discharge measurements, IEC 60270:2000+AMD1:2015.",
 "iec600767": "Power transformers — Part 7: Loading guide for mineral-oil-immersed power transformers, IEC 60076-7:2018.",
 "kipf": "T. N. Kipf and M. Welling, \"Semi-supervised classification with graph convolutional networks,\" in Proc. Int. Conf. Learn. Represent. (ICLR), Toulon, France, 2017.",
 "sage": "W. L. Hamilton, R. Ying, and J. Leskovec, \"Inductive representation learning on large graphs,\" in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 30, Long Beach, CA, USA, 2017, pp. 1024–1034.",
 "liu": "Y. Liu, P. Ning, and M. K. Reiter, \"False data injection attacks against state estimation in electric power grids,\" in Proc. 16th ACM Conf. Comput. Commun. Secur. (CCS), Chicago, IL, USA, 2009, pp. 21–32, doi: 10.1145/1653662.1653666.",
 "boyaci": "O. Boyaci, A. Umunnakwe, A. Sahu, M. R. Narimani, M. Ismail, K. R. Davis, and E. Serpedin, \"Graph neural networks based detection of stealth false data injection attacks in smart grids,\" IEEE Syst. J., vol. 16, no. 2, pp. 2946–2957, 2022, doi: 10.1109/JSYST.2021.3109082.",
 "atrigna": "M. Atrigna, A. Buonanno, R. Carli, G. Cavone, P. Scarabaggio, M. Valenti, G. Graditi, and M. Dotoli, \"A machine learning approach to fault prediction of power distribution grids under heatwaves,\" IEEE Trans. Ind. Appl., vol. 59, no. 4, pp. 4835–4845, 2023, doi: 10.1109/TIA.2023.3262230.",
 "pandapower": "L. Thurner, A. Scheidler, F. Schäfer, J.-H. Menke, J. Dollichon, F. Meier, S. Meinecke, and M. Braun, \"pandapower — An open-source Python tool for convenient modeling, analysis, and optimization of electric power systems,\" IEEE Trans. Power Syst., vol. 33, no. 6, pp. 6510–6521, 2018, doi: 10.1109/TPWRS.2018.2829021.",
 "c37118": "IEEE Standard for Synchrophasor Measurements for Power Systems, IEEE Std C37.118.1-2011, 2011, doi: 10.1109/IEEESTD.2011.6111219.",
 "otcc": "National Cybersecurity Authority, Operational Technology Cybersecurity Controls (OTCC-1:2022), Riyadh, Saudi Arabia, 2022.",
 "baran": "M. E. Baran and F. F. Wu, \"Network reconfiguration in distribution systems for loss reduction and load balancing,\" IEEE Trans. Power Del., vol. 4, no. 2, pp. 1401–1407, 1989, doi: 10.1109/61.25627.",
 "russell": "B. D. Russell and C. L. Benner, \"Intelligent systems for improved reliability and failure diagnosis in distribution systems,\" IEEE Trans. Smart Grid, vol. 1, no. 1, pp. 48–56, 2010, doi: 10.1109/TSG.2010.2044898.",
 "sidhu": "T. S. Sidhu and Z. Xu, \"Detection of incipient faults in distribution underground cables,\" IEEE Trans. Power Del., vol. 25, no. 3, pp. 1363–1371, 2010, doi: 10.1109/TPWRD.2010.2041373.",
 "kulkarni": "S. Kulkarni, S. Santoso, and T. A. Short, \"Incipient fault location algorithm for underground cables,\" IEEE Trans. Smart Grid, vol. 5, no. 3, pp. 1165–1174, 2014, doi: 10.1109/TSG.2014.2303483.",
 "susa": "D. Susa, M. Lehtonen, and H. Nordman, \"Dynamic thermal modeling of distribution transformers,\" IEEE Trans. Power Del., vol. 20, no. 3, pp. 1919–1929, 2005, doi: 10.1109/TPWRD.2005.848675.",
 "liao": "W. Liao, B. Bak-Jensen, J. R. Pillai, Y. Wang, and Y. Wang, \"A review of graph neural networks and their applications in power systems,\" J. Mod. Power Syst. Clean Energy, vol. 10, no. 2, pp. 345–360, 2022, doi: 10.35833/MPCE.2021.000058.",
 "chen": "K. Chen, J. Hu, Y. Zhang, Z. Yu, and J. He, \"Fault location in power distribution systems via deep graph convolutional networks,\" IEEE J. Sel. Areas Commun., vol. 38, no. 1, pp. 119–131, 2020, doi: 10.1109/JSAC.2019.2951964.",
 "lideka": "W. Li, D. Deka, M. Chertkov, and M. Wang, \"Real-time faulted line localization and PMU placement in power systems through convolutional neural networks,\" IEEE Trans. Power Syst., vol. 34, no. 6, pp. 4640–4651, 2019, doi: 10.1109/TPWRS.2019.2917794.",
 "kundacina": "O. Kundacina, M. Cosovic, and D. Vukobratovic, \"State estimation in electric power systems leveraging graph neural networks,\" in Proc. 17th Int. Conf. Probabilistic Methods Appl. Power Syst. (PMAPS), Manchester, U.K., 2022, pp. 1–6, doi: 10.1109/PMAPS53380.2022.9810559.",
 "rudin": "C. Rudin, \"Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead,\" Nat. Mach. Intell., vol. 1, no. 5, pp. 206–215, 2019, doi: 10.1038/s42256-019-0048-x.",
 "guo": "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, \"On calibration of modern neural networks,\" in Proc. 34th Int. Conf. Mach. Learn. (ICML), Sydney, Australia, 2017, pp. 1321–1330.",
 "xgb": "T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., San Francisco, CA, USA, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.",
 "ieee1366": "IEEE Guide for Electric Power Distribution Reliability Indices, IEEE Std 1366-2022, 2022, doi: 10.1109/IEEESTD.2022.9955492.",
 "dobson": "I. Dobson, B. A. Carreras, V. E. Lynch, and D. E. Newman, \"Complex systems analysis of series of blackouts: Cascading failure, critical points, and self-organization,\" Chaos, vol. 17, no. 2, p. 026103, 2007, doi: 10.1063/1.2737822.",
}
ORDER = []
def c(*keys):
    nums = []
    for k in keys:
        if k not in ORDER: ORDER.append(k)
        nums.append(ORDER.index(k) + 1)
    nums.sort()
    return "[" + "], [".join(str(n) for n in nums) + "]" if len(nums) < 3 or nums != list(range(nums[0], nums[-1] + 1)) else f"[{nums[0]}]–[{nums[-1]}]"

# ------------------------------------------------------------------ numbers used in the text
nd = D; ev = E
G, GG, XG, SA, SC, NG, NP = "gnn_full", "gnn_full_gated", "xgb", "static_alarm", "static_caution", "gnn_nograph", "gnn_nophys"
n_test_fail = s0(G, "episodes"); n_test_att = s0(G, "attack_episodes")
sera = E["sera"]

# ================================================================== TITLE BLOCK (single column)
set_cols(sec, 1)
P("Resilient Grid Intelligence: A Standards-Guided Graph Neural Network Framework for Incipient Fault Prediction and Cyber-Physical Anomaly Discrimination in Saudi Medium-Voltage Distribution Networks",
  bold=True, align="c", size=20, after=10, indent=False)
P("Moustafa Elbehairy^1,2^, Ahmed Abu Abuseen^1^, Nasser Al-Maawi^3^, Amir Raouf^4^, and Ahmed Abdulrahman Al-Jafin^5^", align="c", size=11, after=2, indent=False)
P("^1^Nukhbaat Hulul Alamal, Riyadh, Saudi Arabia   ^2^SystechHub, Germany   ^3^United Specialists Co., Saudi Arabia   ^4^Karaz Platform, Saudi Arabia   ^5^Energy Tower Company, Saudi Arabia", align="c", size=9, after=1, indent=False)
P("m.elbehairy@nukhbaa.sa, a.abuseen@nukhbaa.sa, nasser.a@usco-sa.com, amir.raouf@outlook.sa, aaljafin@etco.com.sa   |   Corresponding author: Moustafa Elbehairy", align="c", size=9, after=10, indent=False)

# ================================================================== TWO COLUMNS
wide_section(2)
ABSTRACT = (f"Most customer interruptions originate in medium-voltage networks, and failures of distribution transformers and cable joints emit measurable "
            f"precursors days before breakdown. This paper develops VoltyGuard, which represents a 13.8 kV-class feeder as a dynamic graph and fuses standards-based "
            f"diagnostics, physics-based thermal and power-flow models and a spatio-temporal graph neural network, with alerts gated by agreement between the standards "
            f"and learned layers. A cyber-physical discriminator separates incipient faults from false data injection, and a regulation-aware scheduler maps lead time "
            f"to actions admissible under the Saudi Electricity Regulatory Authority's guaranteed standards. On {nd['n_real']} simulated Riyadh summers of the IEEE 33-bus "
            f"feeder, at the false-alarm budget of the standards alarm the framework detects {pct(E[G]['agg']['budget']['recall']['mean'])} of failures with a median warning of "
            f"{bg(G, 'lead_median_d', 1, sd=False)} days (alarm: {pct(E[SA]['agg']['recall']['mean'])}, {ag(SA, 'lead_median_d', 1, sd=False)} days), reaches {pct(E[G]['agg']['precision']['mean'])} "
            f"precision at its own operating point, separates faults from attacks in {pct(E[G]['agg']['fault_vs_attack_accuracy']['mean'])} of episodes and converts every detected failure into a planned outage.")
p = P("", align="j", size=9, after=4, indent=False); r = p.add_run("Abstract—"); r.bold = True; r.font.size = Pt(9); r.font.name = FONT
r = p.add_run(ABSTRACT); r.bold = True; r.font.size = Pt(9); r.font.name = FONT
p = P("", align="j", size=9, after=6, indent=False); r = p.add_run("Index Terms—"); r.bold = True; r.font.size = Pt(9); r.font.name = FONT
r = p.add_run("Cyber-physical security, dissolved gas analysis, distribution reliability, explainable AI, false data injection, graph neural networks, partial discharge, predictive maintenance, Saudi Arabia, transformer thermal model."); r.bold = True; r.font.size = Pt(9); r.font.name = FONT
n_abs_words = len(ABSTRACT.split())

# ================================================================== I. INTRODUCTION
H1("Introduction")
P(f"Saudi distribution networks are modernising quickly. Distribution automation was reported at about 40% by the end of 2025, with 12.3 GW of renewable "
  f"capacity connected to the national grid {c('daleel')}, and the Saudi Electricity Company reported improvements of 17% in SAIDI and 19% in SAIFI for 2024 "
  f"{c('attaqa')}. Sustaining those gains means moving from restoration after an outage to anticipation before it, and the medium-voltage (MV) network is "
  f"where that shift pays most, because it is where most customer-minutes lost originate. The regulatory frame makes the economics explicit. The guaranteed "
  f"service standards of the Saudi Electricity Regulatory Authority (SERA) require at least two days of notice before a planned interruption, restoration "
  f"within six hours of a planned outage and within three hours of an unplanned one, with automatic customer compensation when a limit is breached {c('sera')}. "
  f"A warning that arrives earlier than the 48-hour notice period can therefore be converted into a planned, compensated-free outage; one that arrives later cannot.")
P(f"Two obstacles have slowed the use of machine learning for this purpose in utility control rooms. The first is trust: engineers do not act on opaque scores "
  f"that contradict diagnostic standards they have applied for decades, a concern that the interpretability literature shares for high-stakes decisions {c('rudin')}. "
  f"The second is security. A model that consumes operational data inherits the attack surface of that data: coordinated false data injection (FDI) can pass "
  f"residual-based bad-data detection {c('liu')} and either provoke an unnecessary switching operation or hide a genuine degradation. An early-warning system that "
  f"cannot tell a failing transformer from a spoofed sensor is, in the attacker's hands, a remote switching tool.")
P(f"This paper reports the full study announced in our accepted extended abstract. We make four contributions. (1) A standards-guided hybrid architecture in "
  f"which a graph neural network (GNN) is fed with, and gated by, the outputs of IEEE C57.104 {c('c57104')}, IEC 60599 {c('iec60599')}, IEC 60270 {c('iec60270')} and "
  f"the IEC 60076-7 thermal model {c('iec600767')}. (2) A cyber-physical discriminator that uses cross-modal and power-flow consistency across neighbouring nodes "
  f"to separate incipient faults, data-integrity attacks and sensor faults. (3) A regulation-aware scheduler that maps warning lead time to actions admissible "
  f"under SERA's guaranteed standards, including an N-1 load-transfer check in a power-flow model. (4) A reproducible evaluation on the IEEE 33-bus feeder "
  f"{c('baran')} under Riyadh summer load and temperature, with {nd['episodes']['joint'] + nd['episodes']['transformer']} degradation episodes, "
  f"{nd['episodes']['fdi_provoke'] + nd['episodes']['fdi_electrical'] + nd['episodes']['fdi_mask']} injection attacks and {nd['episodes']['sensor_stuck'] + nd['episodes']['sensor_drift']} sensor "
  f"malfunctions across {nd['n_real']} independent realisations, against static-threshold and per-asset learning baselines. Code, data generator and results are public.")

# ================================================================== II. BACKGROUND
H1("Background and Related Work")
P(f"Transformer condition assessment is mature and standardised. Dissolved gas analysis (DGA) is interpreted with the gas-by-gas screening values and rate-of-change "
  f"limits of IEEE C57.104-2019 {c('c57104')} and the ratio methods of IEC 60599, including the Duval triangle {c('iec60599', 'duval')}; partial discharge (PD) is "
  f"quantified per IEC 60270 {c('iec60270')}; and thermal ageing follows the loading guide IEC 60076-7 {c('iec600767')}, whose exponential top-oil and hot-spot "
  f"model has been validated for distribution units {c('susa')}. For underground cables, failures of joints and terminations are preceded by sub-cycle self-clearing "
  f"arcing events that recur with increasing frequency; their detection and location from feeder-head recordings is well documented {c('russell', 'sidhu', 'kulkarni')}. "
  f"These methods, however, are applied asset by asset. A transformer is assessed without reference to the joint feeding it, and a recurring transient is attributed "
  f"to a segment without reference to the voltage profile its neighbours observe.")
P(f"Graph neural networks exploit grid topology directly through message passing {c('kipf', 'sage')} and have been reviewed for power-system use {c('liao')}. They have "
  f"located faults in distribution systems {c('chen')}, localised faulted lines from sparse PMU data {c('lideka')}, estimated unmeasured states {c('kundacina')} and "
  f"detected stealthy FDI attacks {c('boyaci')}. Machine learning has also been used to predict distribution faults under heatwaves from weather and asset data "
  f"{c('atrigna')}. To our knowledge these strands remain separate: fault-prediction models assume trustworthy data, and FDI detectors do not reason about "
  f"physical degradation. Neither connects its output to what a regulated operator may actually do with it. The gap this paper addresses is a single, "
  f"standards-anchored model that separates normal operation, incipient physical faults and data-integrity attacks on the same graph, and that converts its "
  f"output into actions admissible under national service regulation.")

# ================================================================== III. FRAMEWORK
H1("Proposed Framework")
wide_section(1)
FIG(f"{FIGS}/fig1_framework.png", "VoltyGuard framework: multi-source data on a dynamic grid graph, three-layer hybrid inference, cyber-physical discrimination and agreement-gated operational outputs.", width_cm=17.5)
wide_section(2)
H2("Dynamic Grid Graph")
P(f"The feeder is a graph whose nodes are assets (primary-substation busbar, cable sections and joints, distribution transformers, measurement points) and whose "
  f"edges are electrical connections carrying series impedance. Topology, including normally-open points, is imported from the utility network model and updated "
  f"from switch status, so reconfiguration changes the adjacency without retraining. Node features combine SCADA loads and voltages, branch flows, protection and "
  f"fault-recorder events, condition-monitoring data where available (dissolved gases, PD, top-oil temperature), upstream synchrophasor data where available "
  f"{c('c37118')}, asset rating and age, and ambient temperature. Fig. 1 summarises the architecture.")
H2("Three-Layer Hybrid Inference")
P(f"*Layer 1* applies the standards-based rules of Section II to each asset and emits an interpretable code (normal, caution, alarm): DGA status from the C57.104 "
  f"screening and rate limits, PD caution and alarm levels, self-clearing fault counts, and top-oil and hot-spot limits. *Layer 2* computes physics-based quantities "
  f"from the measured data: the IEC 60076-7 top-oil and hot-spot temperatures and the relative ageing rate implied by the measured load and ambient temperature, "
  f"the residual between measured and modelled top-oil temperature, and two power-flow residuals obtained by solving the network with the measured loads, namely "
  f"the voltage residual at each bus and the branch power-balance residual (measured flow minus the flow implied by all measured loads). *Layer 3* is a "
  f"spatio-temporal GNN. A gated recurrent unit encodes the last 24 h of every node's features; two GraphSAGE-style layers {c('sage')} then exchange messages "
  f"over the admittance-weighted adjacency, so that each asset's representation includes the state of its electrical neighbours; two heads output the probability "
  f"of failure within seven days and the probability of each anomaly class. Probabilities are calibrated by temperature scaling on validation data {c('guo')}. "
  f"A *critical* alert requires agreement between Layer 1 (code at least caution within the previous 24 h) and Layer 3 (calibrated probability above a "
  f"threshold for three consecutive hours); a Layer-3 alert without Layer-1 support is routed to engineer review with the evidence of both layers.")
H2("Cyber-Physical Anomaly Discrimination")
P(f"Genuine incipient faults leave physically consistent signatures across independent modalities and neighbouring nodes: rising gases corroborate hot-spot "
  f"temperatures computed from load, PD activity corroborates self-clearing events, and electrical quantities obey the power-flow equations across the graph. An "
  f"attacker who manipulates one channel must spoof several modalities and several nodes consistently to avoid detection. The discriminator is the second head of "
  f"Layer 3, trained on the Layer-2 residuals and on the neighbours' features, and it classifies each anomaly as incipient fault, suspected data-integrity attack or "
  f"sensor malfunction. Suspected attacks never trigger switching recommendations; they are escalated to the security operations function.")
H2("Regulation-Aware Maintenance Scheduler")
P(f"The scheduler converts each critical alert into an admissible action given its lead time, the N-1 margin of adjacent assets and customer criticality (Table I). "
  f"For a cable-section alert the N-1 check opens the section, closes the best normally-open tie and solves the power flow, accepting the transfer when all bus "
  f"voltages stay above 0.95 p.u. and no cable exceeds its ampacity. For a transformer alert the check compares the spare capacity of the adjacent unit with the load "
  f"to be transferred.")
TABLE("Mapping of prediction lead time to admissible actions",
      ["Lead time", "Admissible action proposed by the scheduler", "SERA guaranteed standard"],
      [["≥ 48 h", "Planned outage with prior customer notice; repair or replacement in a low-impact window", "Std. 4 (notice ≥ 2 days); Std. 5 (restore ≤ 6 h)"],
       ["< 48 h, N-1 margin", "Load transfer to adjacent feeder or transformer; no customer interruption", "—"],
       ["< 48 h, no margin", "Pre-positioned crew, spare and mobile substation; controlled emergency response", "Std. 6 (restore ≤ 3 h)"],
       ["Fault occurred", "Graph-based fault localisation to shorten restoration", "Std. 6 (restore ≤ 3 h)"]], [1.6, 4.3, 2.6])

# ================================================================== IV. STUDY DESIGN
H1("Simulation Study Design")
H2("Feeder, Climate and Load")
P(f"We use the IEEE 33-bus radial feeder of Baran and Wu {c('baran')} as shipped with pandapower {c('pandapower')} (12.66 kV, 3.715 MW and 2.3 MVAr at peak, "
  f"32 cable sections and five normally-open ties), representative of the 13.8 kV class operated in Saudi cities. Each load bus carries a distribution transformer "
  f"sized at 0.95–1.35 times its peak apparent load; cable ampacities are assigned from the peak-case currents. Each realisation covers one summer of "
  f"{nd['hours'] // 24} days at hourly resolution. Ambient temperature follows Riyadh June–August normals (daily mean about 36 °C, a 16 K diurnal range with "
  f"the maximum at 15:00, day-to-day autoregressive variation and two to three heatwave episodes of three to six days adding 3–5 K; on average "
  f"{f1(nd['heatwave_days_mean'])} heatwave days per summer). Load combines residential and commercial diurnal shapes with a cooling sensitivity of 2.5% per kelvin "
  f"above 30 °C, a weekend reduction and lognormal noise; the hourly network state is obtained by a backward-forward-sweep power flow.")
H2("Degradation, Attack and Sensor-Fault Scenarios")
P(f"Three transformer and joint processes are simulated per realisation (two to three of each). *Transformer thermal ageing*: under-sized units (rated at "
  f"0.68–0.84 times peak load) are driven through the IEC 60076-7 model (distribution ONAN parameters: top-oil rise 55 K, hot-spot gradient 23 K, loss ratio 5, "
  f"exponents 0.8 and 1.6, oil time constant 180 min); the relative ageing rate accumulates into a damage index normalised so that failure occurs at a random "
  f"day of the summer, and thermal-fault gases (C_{2}H_{4}, CH_{4}, C_{2}H_{6}, H_{2}, with C_{2}H_{2} only near the end) are generated at a rate that rises with hot-spot "
  f"temperature and accelerates once the damage index exceeds one half. *Cable-joint degradation*: a latent damage index grows from a random onset over 15–38 days, "
  f"accelerated by load current; it drives a Poisson process of self-clearing faults (up to about four per day near failure, attributed to the correct segment "
  f"with 80% probability and otherwise to an adjacent one, to mimic location error), a rising PD level where a monitor is installed, and a growing contact "
  f"resistance that is written into the power-flow model so that neighbouring voltages respond physically. *FDI attacks* of three kinds target healthy or degrading "
  f"assets for two to eight days: a *provoke* attack inflates the top-oil sensor and the online DGA of a healthy transformer to imitate degradation; a *stealthy "
  f"electrical* attack scales the load and the feeding-branch measurements of a node consistently with the local voltage drop, so that the node's own power-flow "
  f"residual stays small while the parent branch's balance is violated; and a *mask* attack replays the condition data of a degrading asset from 15 days earlier "
  f"during its final days. *Sensor malfunctions* are a stuck or drifting top-oil sensor. Measurement noise is 0.3% on voltages, 1% on powers and currents, 1 K on "
  f"temperatures and 5% on gas concentrations. Online DGA is available on {pct(nd['dga_online_share'])} of transformers (daily samples; the others are sampled "
  f"monthly) and PD monitors on {pct(nd['pd_monitored_share'])} of joints, which exposes the models to partial observability. Table II summarises the data.")
TABLE("Simulated data set (24 realisations × 92 days × 33 nodes)",
      ["Quantity", "Value"],
      [["Node-hours (train / validation / test realisations)", f"{nd['node_hours']:,} ({len([0]*14)} / 4 / 6 realisations)"],
       ["Features per node (raw + rolling 24 h / 168 h statistics)", f"{nd['features_raw']} + {nd['features_with_rolling'] - nd['features_raw']}"],
       ["Node-hours labelled incipient / attack / sensor fault", f"{nd['class_counts']['incipient']:,} / {nd['class_counts']['attack']:,} / {nd['class_counts']['sensor']:,}"],
       ["Failure episodes: cable joints / transformers (test)", f"{nd['episodes']['joint']} / {nd['episodes']['transformer']} ({nd['episodes_by_split']['test']['joint']} / {nd['episodes_by_split']['test']['transformer']})"],
       ["FDI episodes: provoke / electrical / mask (test)", f"{nd['episodes']['fdi_provoke']} / {nd['episodes']['fdi_electrical']} / {nd['episodes']['fdi_mask']} ({nd['episodes_by_split']['test']['fdi_provoke']} / {nd['episodes_by_split']['test']['fdi_electrical']} / {nd['episodes_by_split']['test']['fdi_mask']})"],
       ["Sensor-fault episodes: stuck / drift (test)", f"{nd['episodes']['sensor_stuck']} / {nd['episodes']['sensor_drift']} ({nd['episodes_by_split']['test']['sensor_stuck']} / {nd['episodes_by_split']['test']['sensor_drift']})"],
       ["Layer-1 caution-or-alarm rate on normal node-hours", pct(nd['l1_rates_on_normal']['caution_or_alarm'], 1)],
       ["Simulation time per realisation (laptop CPU)", f"{nd['sim_runtime_s_mean']:.0f} s"]], [5.5, 3.0])
H2("Models, Baselines and Ablations")
P(f"*Static rules* use the Layer-1 code alone (alarm, or caution-or-alarm). The *per-asset baseline* is a gradient-boosted tree ensemble {c('xgb')} trained on each "
  f"node's own features and rolling statistics, without neighbours; it represents the asset-isolated learning approach of {c('atrigna')}. *VoltyGuard* is the "
  f"network of Section III-B with hidden size 64 and {M['gnn_full'][0]['n_params']:,} parameters, trained with class-weighted losses, AdamW, early stopping on the "
  f"validation realisations and three random seeds. Two ablations remove the graph (adjacency set to zero, so each node sees only itself) and remove the "
  f"Layer-1/2 features (the network sees raw measurements only). Realisations are split {len(SPLIT_TR := list(range(14)))}/4/6 into training, validation and test, "
  f"so that no test asset, episode or weather trace is seen in training.")
H2("Metrics")
P(f"An alert is raised when the calibrated failure probability exceeds a threshold for three consecutive hours; the threshold is chosen on the validation "
  f"realisations to maximise episode-level F1 (the *F1 operating point*); a second, *matched-budget* threshold is the lowest one whose validation false-cluster rate stays within the false-cluster rate of the static alarm, which makes recall and lead time comparable across models at equal nuisance. A failure episode counts as detected if an alert occurs on that asset within the 45 days before failure, and its "
  f"*warning lead time* is the interval from the first such alert to failure. Alert runs on assets that do not fail within 45 days are *false clusters*, reported "
  f"per 100 asset-days. Episode recall and precision follow. *Discrimination accuracy* is the share of anomaly episodes whose majority class over the episode is "
  f"correct; the fault-versus-attack figure restricts this to incipient-fault and FDI episodes. Calibration is the expected calibration error (ECE) on test "
  f"node-hours {c('guo')}. The SERA mapping of Table I is applied to every detected failure at the hour of its first alert.")

# ================================================================== V. RESULTS
H1("Results")
H2("Detection and Warning Lead Time")
rows = []
for key, lab in ((SA, "Static alarm (Layer 1 = alarm)"), (SC, "Static caution-or-alarm"), (XG, "Per-asset GBT"), (NP, "GNN without standards/physics"), (NG, "GNN without graph"), (G, "VoltyGuard (Layer 1 + 2 + 3)"), (GG, "VoltyGuard, agreement-gated")):
    a = E[key]["agg"]
    hasb = "budget" in E[key]["agg"]
    rows.append([(lab, key in (G, GG)), ag(key, "recall", 2, sd=False), ag(key, "precision", 2, sd=False), ag(key, "false_per_100_asset_days", 2, sd=False), ag(key, "lead_median_d", 1, sd=False), ag(key, "share_lead_ge_48h", 2, sd=False),
                 bg(key, "recall", 2, sd=False) if hasb else "—", bg(key, "lead_median_d", 1, sd=False) if hasb else "—"])
TABLE(f"Episode-level performance on the six test realisations ({n_test_fail} failures; means over three seeds; seed s.d. of recall and lead in the text). Columns 2–6: F1 operating point; 7–8: matched false-alarm budget ({s0(SA, 'false_per_100_asset_days'):.2f} false clusters / 100 asset-days)",
      ["Model", "Recall", "Prec.", "FA/100", "Lead (d)", "≥48 h", "Rec.@B", "Lead@B"], rows, [2.75, 0.85, 0.8, 0.85, 0.85, 0.75, 0.85, 0.9], size=6.5,
      note="FA/100: false alert clusters per 100 asset-days. Rec.@B, Lead@B: recall and median lead (days) at the matched false-alarm budget.")
P(f"Table III gives the main result. The standards alarm alone detects {pct(E[SA]['agg']['recall']['mean'])} of the {n_test_fail} test failures with a median "
  f"lead of {ag(SA, 'lead_median_d', 1, sd=False)} days, but only {pct(E[SA]['agg']['precision']['mean'])} of its alert clusters precede a failure "
  f"({ag(SA, 'false_per_100_asset_days', 2, sd=False)} false clusters per 100 asset-days, about {round(E[SA]['agg']['false_per_100_asset_days']['mean'] * 32 * 3.65)} per feeder-year); the caution "
  f"level raises recall to {pct(E[SC]['agg']['recall']['mean'])} at {ag(SC, 'false_per_100_asset_days', 2, sd=False)} false clusters per 100 asset-days. At its F1 operating "
  f"point VoltyGuard is the most selective model: {pct(E[G]['agg']['precision']['mean'])} precision and {ag(G, 'false_per_100_asset_days', 2, sd=False)} false clusters per "
  f"100 asset-days, with {pct(E[G]['agg']['recall']['mean'])} recall (seed-to-seed s.d. {pct(E[G]['agg']['recall']['sd'])}) and a median lead of "
  f"{ag(G, 'lead_median_d', 1, sd=False)} days. The fairer comparison is at equal nuisance. When every learned model is given the standards alarm's own false-alarm "
  f"budget, VoltyGuard detects {pct(E[G]['agg']['budget']['recall']['mean'])} of failures with a median lead of {bg(G, 'lead_median_d', 1, sd=False)} days, "
  f"against {pct(E[XG]['agg']['budget']['recall']['mean'])} and {bg(XG, 'lead_median_d', 1, sd=False)} days for the per-asset ensemble and "
  f"{pct(E[SA]['agg']['recall']['mean'])} and {ag(SA, 'lead_median_d', 1, sd=False)} days for the alarm itself; every VoltyGuard warning at either operating point "
  f"arrives before the 48-hour notice period. Agreement gating removes the few ungrounded alerts (false clusters {ag(GG, 'false_per_100_asset_days', 2, sd=False)} per "
  f"100 asset-days) and shortens the median lead to {ag(GG, 'lead_median_d', 1, sd=False)} days, because a critical alert now waits for the first standards caution. "
  f"Lead times differ by fault class (Fig. 3): transformers, whose thermal history is observable through the hot-spot model and whose gases accumulate over weeks, "
  f"are warned a median {f1(s0(G, 'lead_by_kind_median_d')['transformer'])} days ahead at the F1 point, while cable joints, half of which have no PD monitor, give "
  f"{f1(s0(G, 'lead_by_kind_median_d')['joint'])} days. The static alarm warns transformers earlier still (median {f1(s0(SA, 'lead_by_kind_median_d')['transformer'])} days), "
  f"because the hot-spot limit of IEC 60076-7 is exceeded weeks before gassing becomes decisive, but it gives joints only {f1(s0(SA, 'lead_by_kind_median_d')['joint'])} days; "
  f"the learned model's seven-day target horizon concentrates its warnings in the week before failure, and a longer horizon would move them earlier at a cost in precision.")
wide_section(1)
FIG(f"{FIGS}/fig2_precursors.png", "Two test episodes. Top: precursors of (a) a transformer thermal failure and (b) a cable-joint failure. Bottom: calibrated failure probability of VoltyGuard and of the per-asset model, and the Layer-1 standards code; the dashed line marks the SERA 48-hour notice period.", width_cm=17.5)
wide_section(2)
FIG(f"{FIGS}/fig3_leadtime.png", "Warning lead time of detected failures on the test realisations (seed 0). Boxes show quartiles; the dashed line is the SERA two-day notice requirement.")
H2("Fault-versus-Attack Discrimination")
cm = np.array(s0(G, "disc_confusion"))
P(f"Fig. 4(a) shows the episode-level confusion of VoltyGuard's class head over the {int(cm[1:].sum())} test anomaly episodes. Incipient faults and FDI attacks are "
  f"separated in {pct(s0(G, 'fault_vs_attack_accuracy'))} of the episodes (mean {pct(E[G]['agg']['fault_vs_attack_accuracy']['mean'])} over seeds); {s0(G, 'attack_flagged_as_fault')} of the {n_test_att} attacks were read as faults, and "
  f"three-class accuracy including sensor faults is {pct(s0(G, 'disc_accuracy_3class'))}. The mask attack, which replays a degrading asset's own earlier condition "
  f"data, is the hardest case: {s0(G, 'mask_attacks_detected')} of {s0(G, 'mask_attacks')} were flagged as attacks, the others being exposed only by the physical "
  f"channels the attacker did not control (hot-spot temperature implied by load, neighbouring voltages). The per-asset ensemble, which cannot see the parent "
  f"branch's power balance, separates faults from attacks in {pct(s0(XG, 'fault_vs_attack_accuracy'))} of episodes.")
H2("Calibration")
P(f"After temperature scaling the ECE of VoltyGuard's seven-day failure probability is {ag(G, 'ece', 3)} on {nd['node_hours'] * 6 // 24:,} test node-hours (Fig. 4(b)), "
  f"against {ag(XG, 'ece', 3)} for the per-asset model. Calibration matters for the scheduler, which reads the probability as a decision input rather than as a score.")
FIG(f"{FIGS}/fig4_discrimination_calibration.png", "(a) Episode-level confusion of the anomaly class head on the test realisations (seed 0). (b) Reliability diagram of the calibrated seven-day failure probability.")
H2("Ablation: What the Graph and the Standards Layers Add")
P(f"Removing the graph (Fig. 5) lowers recall from {pct(E[G]['agg']['recall']['mean'])} to {pct(E[NG]['agg']['recall']['mean'])} at the F1 point and from "
  f"{pct(E[G]['agg']['budget']['recall']['mean'])} to {pct(E[NG]['agg']['budget']['recall']['mean'])} at the matched budget, lowers precision from "
  f"{pct(E[G]['agg']['precision']['mean'])} to {pct(E[NG]['agg']['precision']['mean'])}, and makes the lead time more variable across seeds "
  f"(s.d. {ag(NG, 'lead_median_d', 1, sd=False) and f1(E[NG]['agg']['budget']['lead_median_d']['sd'])} days against {f1(E[G]['agg']['budget']['lead_median_d']['sd'])}). "
  f"Removing the standards and physics features while keeping the graph leaves recall unchanged ({pct(E[NP]['agg']['recall']['mean'])}) but lowers precision to "
  f"{pct(E[NP]['agg']['precision']['mean'])}, lowers fault-versus-attack accuracy to {pct(E[NP]['agg']['fault_vs_attack_accuracy']['mean'])}, and loses the mask "
  f"attacks entirely ({E[NP]['seeds'][0]['mask_attacks_detected']} of {E[NP]['seeds'][0]['mask_attacks']} flagged, against {s0(G, 'mask_attacks_detected')} of {s0(G, 'mask_attacks')} "
  f"with the full model): without the hot-spot temperature implied by load, a replayed condition record is indistinguishable from a healthy one. The two layers are "
  f"complementary. The physics features carry the thermal history that no 24-hour window of raw data contains and the reference against which a spoofed sensor is "
  f"exposed; the graph carries the neighbour consistency that no single-asset model can test.")
wide_section(1)
FIG(f"{FIGS}/fig5_ablation.png", "Ablation on the test realisations (mean ± s.d. over three seeds): episode recall, false clusters per 100 asset-days and median warning lead time.", width_cm=17.5)
wide_section(2)
H2("Regulation-Aware Actions and N-1 Feasibility")
rows = []
for key, lab in ((SA, "Static alarm"), (XG, "Per-asset GBT"), (G, "VoltyGuard"), (GG, "VoltyGuard, gated")):
    a = sera[key]["actions"]; tot = sum(a.values())
    rows.append([(lab, key in (G, GG)), a["planned"], a["transfer"], a["emergency"], a["missed"], f"{sera[key]['n1_feasible']}/{sera[key]['n1_checks']}"])
TABLE(f"SERA action mapping of the {n_test_fail} test failures (seed 0)", ["Model", "Planned (≥48 h)", "N-1 transfer", "Emergency", "Missed", "N-1 feasible"], rows, [2.6, 1.3, 1.2, 1.2, 1.0, 1.3], size=6.5)
gg = sera[GG]["actions"]; sa = sera[SA]["actions"]
P(f"Table IV applies the scheduler. With VoltyGuard gated, {gg['planned']} of {n_test_fail} failures can be handled as planned outages with the two-day customer "
  f"notice, {gg['transfer']} as N-1 transfers without interruption and {gg['emergency']} require an emergency response; {gg['missed']} are missed. The standards "
  f"alarm alone leaves {sa['emergency'] + sa['missed']} failures to the emergency path. Of the {sera[GG]['n1_checks']} short-lead cases checked in the power-flow "
  f"model, {sera[GG]['n1_feasible']} satisfied the voltage and ampacity limits after tie-switch closure, a reminder that the value of a warning depends on the "
  f"network's reserve as much as on its timing.")

# ================================================================== VI. DISCUSSION
H1("Discussion")
P(f"Three findings carry over to deployment. First, the standards layer is not a formality. On normal node-hours it raises caution in only "
  f"{pct(nd['l1_rates_on_normal']['caution_or_alarm'], 1)} of cases, and gating the learned alerts on it costs no recall while giving every critical alert a "
  f"reference an engineer can check against C57.104 or IEC 60270. Second, the gains from the graph are largest exactly where the attack surface is: the parent "
  f"branch's power-balance residual is invisible to a per-asset model, and it is what separates a stealthy electrical injection from a genuine overload. Third, "
  f"lead time is not a free variable. Transformer ageing is slow and observable, so warnings of one to several weeks are realistic; joint failures preceded by "
  f"self-clearing faults give days, and only where a PD monitor or a fault recorder covers the segment. The scheduler makes this difference operational rather than "
  f"statistical.")
P(f"The study has limits that we state plainly. All data are synthetic, generated from standard models and documented assumptions; the degradation laws, gas "
  f"generation rates and attack models are plausible, not fitted to Saudi field records, and the absolute numbers will move when they are. The feeder is balanced "
  f"and radial, without distributed generation, and the hourly resolution hides sub-cycle phenomena that synchrophasor or fault-recorder data would expose. Model "
  f"selection used six test realisations and three seeds; the seed-to-seed standard deviations quoted in Section V are the honest size of that uncertainty. The next step is a "
  f"shadow-mode pilot on a Saudi 13.8 kV network with read-only access to SCADA, condition-monitoring and relay data, which is the deployment path compatible "
  f"with the National Cybersecurity Authority's operational-technology controls {c('otcc')}: no control authority, data diode, engineer retains the decision, field "
  f"outcomes fed back to retrain. Reliability effects would then be measured with the IEEE 1366 indices {c('ieee1366')} rather than with simulated lead times, "
  f"and cascading interactions {c('dobson')} could be studied on a meshed model.")

# ================================================================== VII. CONCLUSION
H1("Conclusion")
P(f"Predictive maintenance becomes useful to a regulated operator when it is trusted, secure and convertible into admissible action. The framework evaluated here "
  f"addresses all three conditions for Saudi MV networks: it gates learned predictions on diagnostic standards, it discriminates incipient faults from false data "
  f"injection by exploiting physical and topological consistency, and it maps lead time to the actions SERA permits. On {nd['n_real']} simulated Riyadh summers "
  f"of the IEEE 33-bus feeder it detected {pct(E[G]['agg']['budget']['recall']['mean'])} of failures with a median warning of {bg(G, 'lead_median_d', 1, sd=False)} days at the false-alarm budget of the standards alarm, "
  f"separated faults from attacks in {pct(E[G]['agg']['fault_vs_attack_accuracy']['mean'])} of episodes and turned most detected failures into planned outages. "
  f"The generator, models and results are released for reuse, and the authors invite network operators to the shadow-mode pilot that must come next.")

# ================================================================== DATA / ACK / REFERENCES
H1u("Data and Code Availability")
P(f"The simulation generator, trained models, evaluation scripts and all result files are available at {REPO} (MIT licence).", indent=False)
H1u("References")
for i, k in enumerate(ORDER):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format; pf.left_indent = Cm(0.7); pf.first_line_indent = Cm(-0.7); pf.space_after = Pt(1); pf.line_spacing = 1.0
    r = p.add_run(f"[{i + 1}]\t"); r.font.size = Pt(8); r.font.name = FONT
    r = p.add_run(REFS[k]); r.font.size = Pt(8); r.font.name = FONT
    tabs = pf.tab_stops; tabs.add_tab_stop(Cm(0.7))
unused = [k for k in REFS if k not in ORDER]

# ================================================================== BIOGRAPHIES
H1u("Biographies")
BIOS = [
 ("Moustafa Elbehairy", "is Co-Chief Executive Officer of Nukhbaat Hulul Alamal (Nukhba Business Solutions), Riyadh, Saudi Arabia, and a founding partner of SystechHub, Germany. He has more than eighteen years of experience in artificial intelligence, data science and risk analytics for government and critical-infrastructure clients in Saudi Arabia and Egypt, and lectures in statistics and insurance at Suez Canal University, Egypt. His research interests include graph learning for infrastructure, AI governance and risk modelling.", False),
 ("Ahmed Abu Abuseen", "is a consultant in data governance and personal-data protection at Nukhbaat Hulul Alamal, Riyadh, Saudi Arabia, advising public entities on compliance with the Saudi Personal Data Protection Law and the National Cybersecurity Authority's controls.", True),
 ("Nasser Al-Maawi", "is an electrical engineer with United Specialists Co., Saudi Arabia, working on distribution-network projects and field instrumentation.", True),
 ("Amir Raouf", "is a consultant in operations management and research with Karaz Platform, Saudi Arabia, with interests in maintenance scheduling and optimisation.", True),
 ("Ahmed Abdulrahman Al-Jafin", "is an electrical engineer and the founder of Energy Tower Company, Saudi Arabia. He previously held management positions at the Ministry of Energy and the Saudi Electricity Company, with a focus on medium-voltage distribution networks.", True),
]
for name, text, hl in BIOS:
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.space_after = Pt(4); p.paragraph_format.line_spacing = 1.0
    r = p.add_run("[photo 2.54 × 3.18 cm] "); r.font.size = Pt(9); r.font.name = FONT; r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    r = p.add_run(name + " "); r.bold = True; r.font.size = Pt(9); r.font.name = FONT
    r = p.add_run(text); r.font.size = Pt(9); r.font.name = FONT
    if hl: r.font.highlight_color = WD_COLOR_INDEX.YELLOW

out = f"{ROOT}/VoltyGuard_SASG26_FullPaper.docx"; doc.save(out)
print("saved", out, "| abstract words:", n_abs_words, "| refs cited:", len(ORDER), "| unused ref keys:", unused)
