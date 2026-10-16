"""AAROGYA MARG - Streamlit frontend (Tier 1 prototype).
PROTOTYPE FOR COURSEWORK. NOT A MEDICAL DEVICE. A CLINICIAN DECIDES. SYNTHETIC DATA ONLY."""
import io
import json
import logging
import os
import time
import uuid
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

import triage_core as tc
from store import Store

st.set_page_config(page_title="AAROGYA MARG", page_icon="🩺", layout="wide")

# ----------------------------------------------------------------- languages
# Labels for Telugu and Hindi are a first draft and MUST be reviewed by a native speaker / clinician.
T = {
    "en": {"title": "AAROGYA MARG", "tag": "Rural triage and referral support", "run": "Run triage",
           "action": "Recommended action", "notice": "PROTOTYPE for coursework. Not a medical device. "
           "A clinician decides. Synthetic data only. Do not enter real patient details.",
           "treat_locally": "Treat locally", "one_more_test": "Do one more test", "refer": "Refer",
           "refer_urgently": "Refer urgently"},
    "te": {"title": "ఆరోగ్య మార్గ్", "tag": "గ్రామీణ ట్రయేజ్ మరియు రిఫరల్ సహాయం", "run": "ట్రయేజ్ నడపండి",
           "action": "సిఫార్సు చేసిన చర్య", "notice": "ఇది కోర్సు ప్రోటోటైప్. వైద్య పరికరం కాదు. నిర్ణయం వైద్యునిదే. "
           "కృత్రిమ డేటా మాత్రమే.",
           "treat_locally": "స్థానికంగా చికిత్స", "one_more_test": "మరో పరీక్ష చేయండి", "refer": "రిఫర్ చేయండి",
           "refer_urgently": "వెంటనే రిఫర్ చేయండి"},
    "hi": {"title": "आरोग्य मार्ग", "tag": "ग्रामीण ट्राइएज और रेफरल सहायता", "run": "ट्राइएज चलाएँ",
           "action": "सुझाई गई कार्रवाई", "notice": "यह कोर्स प्रोटोटाइप है। चिकित्सा उपकरण नहीं है। निर्णय चिकित्सक का होगा। "
           "केवल कृत्रिम डेटा।",
           "treat_locally": "स्थानीय स्तर पर उपचार", "one_more_test": "एक और जाँच करें", "refer": "रेफर करें",
           "refer_urgently": "तुरंत रेफर करें"},
}


def tr(k):
    lang = st.session_state.get("lang", "en")
    return T[lang].get(k, T["en"].get(k, k))


# ----------------------------------------------------------------- resources
@st.cache_resource
def resources():
    os.makedirs("logs", exist_ok=True)
    if not any(isinstance(x, logging.FileHandler) for x in tc.log.handlers):
        tc.log.addHandler(logging.FileHandler("logs/app.log"))  # structured, never contains findings
    tc.log.setLevel(logging.INFO)
    model = tc.load_model("model/model.json")  # refuses to load if SHA-256 does not match
    return model, tc.make_demo_graph(), Store("aarogya.db")


try:
    MODEL, GRAPH, STORE = resources()
except Exception as e:  # fail safe, never a blank screen
    st.error("Model could not be loaded (" + type(e).__name__ + "). Use the paper guideline checklist and "
             "refer for clinician review.")
    st.stop()

ORIGINS = [(i, n) for i, n, ty, x, y in tc.DEMO_FACILITIES if ty in ("SC", "PHC")]
NAME = {i: n for i, n, *_ in tc.DEMO_FACILITIES}
TRI = ["Not asked", "No", "Yes"]
TRI_MAP = {"Not asked": None, "No": False, "Yes": True}
SY_KEYS = list(tc.SYMPTOMS)

SCENARIOS = {
    "A. Lakshmi (54 F): high BP and sugar with chest discomfort": {
        "in_age": 54, "in_sex": "F", "in_sbp": 168, "in_dbp": 102, "in_glu": 240, "in_glu_type": "random",
        "sy_headache": "Yes", "sy_tiredness": "Yes", "sy_chest_discomfort": "Yes"},
    "B. Ravi (62 M): severe hypertension (red flag)": {
        "in_age": 62, "in_sex": "M", "in_sbp": 192, "in_dbp": 118, "in_glu": 130, "in_glu_type": "random",
        "sy_headache": "Yes"},
    "C. Sunita (35 F): healthy screening": {
        "in_age": 35, "in_sex": "F", "in_sbp": 116, "in_dbp": 74, "in_glu": 92, "in_glu_type": "fasting",
        "in_bmi": 22.0, "in_smoker": "No", "in_family": "No", "sy_headache": "No", "sy_polyuria": "No",
        "sy_tiredness": "No", "sy_chest_discomfort": "No", "sy_breathlessness": "No"},
    "D. Bonus: sparse data (age and tiredness only)": {"in_age": 50, "sy_tiredness": "Yes"},
}
NUM_KEYS = ["in_age", "in_sbp", "in_dbp", "in_glu", "in_bmi", "in_hba1c"]


def load_scenario(name):
    ss = st.session_state
    for k in NUM_KEYS:
        ss[k] = None
    ss["in_sex"], ss["in_glu_type"], ss["in_ecg"] = "Not recorded", "random", "Not done"
    ss["in_smoker"] = ss["in_family"] = "Not asked"
    for s in SY_KEYS:
        ss["sy_" + s] = "Not asked"
    for k, v in SCENARIOS[name].items():
        ss[k] = v
    ss["loaded"] = name
    ss.pop("res", None)
    ss.pop("aid", None)


def collect_findings():
    ss = st.session_state

    def num(k):
        v = ss.get(k)
        return None if v is None else float(v)

    f = {"age": num("in_age"), "sex": {"F": "F", "M": "M", "O": "O"}.get(ss.get("in_sex")),
         "sbp_mmhg": num("in_sbp"), "dbp_mmhg": num("in_dbp"), "glucose_mg_dl": num("in_glu"),
         "glucose_type": ss.get("in_glu_type") if num("in_glu") is not None else None,
         "bmi_kg_m2": num("in_bmi"), "smoker": TRI_MAP[ss.get("in_smoker", "Not asked")],
         "family_history": TRI_MAP[ss.get("in_family", "Not asked")],
         "symptoms": {s: TRI_MAP[ss.get("sy_" + s, "Not asked")] for s in SY_KEYS},
         "hba1c_pct": num("in_hba1c"),
         "ecg": {"Normal": "normal", "Abnormal": "abnormal"}.get(ss.get("in_ecg"))}
    return {k: v for k, v in f.items() if v is not None}


# ------------------------------------------------------------------- figures
def prob_fig(res, title=""):
    fig, ax = plt.subplots(figsize=(6.2, 2.3))
    names = [tc.COND_LABEL[c] for c in tc.TARGETS]
    med = np.array([res["probabilities"][c]["median"] for c in tc.TARGETS])
    lo = np.array([res["probabilities"][c]["low80"] for c in tc.TARGETS])
    hi = np.array([res["probabilities"][c]["high80"] for c in tc.TARGETS])
    cols = ["#c0392b" if m >= 0.5 else "#e67e22" if m >= 0.3 else "#27ae60" for m in med]
    y = np.arange(3)
    ax.barh(y, med, color=cols, height=0.55)
    ax.errorbar(med, y, xerr=[med - lo, hi - med], fmt="none", ecolor="black", capsize=3)
    for i in range(3):
        ax.text(min(hi[i] + 0.02, 1.0), y[i], f"{med[i]:.0%} (80%: {lo[i]:.0%}-{hi[i]:.0%})", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.6)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def map_fig(plan, origin):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for a, lst in GRAPH.adj.items():
        for b, _, _ in lst:
            if a < b:
                A, B = GRAPH.nodes[a], GRAPH.nodes[b]
                ax.plot([A["x"], B["x"]], [A["y"], B["y"]], color="#cccccc", lw=0.8, zorder=1)
    style = {"SC": ("o", 60, "#7fb3d5"), "PHC": ("s", 90, "#52be80"), "CHC": ("D", 120, "#f5b041"),
             "DH": ("*", 300, "#e74c3c")}
    for nid, n in GRAPH.nodes.items():
        m, s, c = style[n["type"]]
        if not n["open"]:
            ax.scatter(n["x"], n["y"], marker="X", s=s + 40, color="#555555", zorder=3)
        else:
            ax.scatter(n["x"], n["y"], marker=m, s=s, color=c, edgecolor="black", zorder=3)
        tag = n["name"] + ("" if n["open"] else " (CLOSED)")
        ax.text(n["x"] + 0.8, n["y"] + 0.8, tag, fontsize=7)
    if plan and plan.get("ok"):
        pts = [(GRAPH.nodes[p]["x"], GRAPH.nodes[p]["y"]) for p in plan["path"]]
        ax.plot(*zip(*pts), color="#c0392b", lw=3, zorder=2, label="A* route")
    o = GRAPH.nodes[origin]
    ax.scatter(o["x"], o["y"], marker="P", s=220, color="#8e44ad", edgecolor="black", zorder=4, label="You are here")
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Facility graph (no map tiles; works offline). X = closed", fontsize=9)
    fig.tight_layout()
    return fig


def slip_pdf(res, aid, origin, plan):
    buf = io.BytesIO()
    lines = [
        "AAROGYA MARG - REFERRAL SLIP (PROTOTYPE, SYNTHETIC DATA)", "",
        f"Assessment: {aid}    Model: {res['model_version']}    Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"From: {NAME[origin]}", "",
        f"Recommended action: {tc.ACTION_LABEL[res['recommended_action']]}   (risk: {res['risk_label']})",
        f"Summary: {res['summary']}", ""]
    for f in res.get("red_flags", []):
        lines.append(f"RED FLAG {f['code']}: {f['message']}")
    for c in tc.TARGETS:
        if c in res.get("probabilities", {}):
            p = res["probabilities"][c]
            lines.append(f"{tc.COND_LABEL[c]}: {p['median']:.0%} (80% interval {p['low80']:.0%}-{p['high80']:.0%})")
    if plan and plan.get("ok"):
        lines += ["", f"Refer to: {plan['destination_name']}  ({plan['minutes']:.0f} min, {plan['km']:.0f} km)",
                  "Needs: " + ", ".join(plan["required"])]
    lines += ["", "Accepted / overridden by: ____________________   Signature: ____________",
              "", tc.DISCLAIMER]
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, "\n".join(lines), va="top", fontsize=10, family="monospace", wrap=True)
        pdf.savefig(fig)
        plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------- sidebar
ss = st.session_state
with st.sidebar:
    ss["lang"] = {"English": "en", "తెలుగు (draft)": "te", "हिन्दी (draft)": "hi"}[
        st.selectbox("Language", ["English", "తెలుగు (draft)", "हिन्दी (draft)"])]
    st.markdown("### Facility")
    origin = st.selectbox("You are at", [i for i, _ in ORIGINS], format_func=lambda i: NAME[i])
    closed, no_caps, status_ts = STORE.current_status()
    GRAPH.apply_status(closed, no_caps)
    caps = GRAPH.nodes[origin]["caps"]
    st.caption("Available here: " + ", ".join(sorted(caps)))
    st.markdown("### Value judgements (utility)")
    m_w = st.slider("Missed-case weight", 1, 50, 10, help="How much worse is missing a sick patient than an unneeded referral")
    u_w = st.slider("Unneeded-referral weight", 0.5, 5.0, 1.0, 0.5)
    st.markdown("### Safety")
    kill = st.checkbox("Kill switch: use plain guideline checklist (model off)")
    st.caption("Role: health worker (demo). Model " + tc.MODEL_VERSION)

st.title(f"🩺 {tr('title')}")
st.caption(tr("tag") + "  ·  आरोग्य मार्ग")
st.warning(tr("notice"))

tabs = st.tabs(["1 Assess", "2 What-if", "3 Why", "4 Referral", "5 Decision & log", "6 Evaluation",
                "7 Scenarios", "8 About & limits"])

cost = {"missed_case_weight": m_w, "unneeded_referral_weight": u_w}


def triage(findings, rid=None):
    return tc.run_triage(MODEL, findings, cost_settings=cost, caps=caps, graph=GRAPH, origin=origin,
                         kill_switch=kill, request_id=rid or uuid.uuid4().hex[:8])


def show_result(res):
    if res.get("status") == "invalid":
        for e in res["errors"]:
            st.error(e)
        return
    for f in res["red_flags"]:
        st.error(f"🚨 RED FLAG {f['code']}: {f['message']} → refer urgently. Rules run before the model.")
    if res["degraded"]:
        st.error("The engine could not complete. Refer for clinician review.")
    for w in res["warnings"]:
        st.warning(w)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("action"), f"{res['risk_icon']} {tr(res['recommended_action'])}")
    c2.metric("Risk level", f"{res['risk_icon']} {res['risk_label']}")
    c3.metric("Confidence", str(res["confidence"]).upper())
    c4.metric("Latency", f"{res['latency_ms']:.0f} ms")
    st.info(res["summary"])
    if res["mode"] == "guideline_checklist":
        st.warning("Kill switch is ON. Model not used.")
        return
    if res["insufficient_evidence"] and not res["degraded"]:
        st.warning("Insufficient evidence: too few findings or too wide an interval. 'Treat locally' is blocked until more "
                   "findings are collected.")
    if res.get("probabilities"):
        st.pyplot(prob_fig(res, "Probability with 80% interval (colour + number + label)"))
        plt.close("all")
        st.caption(f"Action agreement across {MODEL.K} model samples: {res['action_agreement']:.0%}. "
                   f"Findings used: {res['n_findings']}.")
    nt = res.get("next_test")
    if res.get("recommended_action") == "refer_urgently" or res.get("safety_notice"):
        st.error("Urgent referral takes priority. Do not delay referral for routine tests or this prototype's output.")
    elif nt:
        st.success(f"Best next test here: **{nt['label']}** (value per cost {nt['voi_per_cost']:.2f}).")
    st.caption(tc.DISCLAIMER)


# --------------------------------------------------------------- 1 Assess
with tabs[0]:
    st.subheader("Patient findings (every field is optional)")
    a, b, c = st.columns(3)
    with a:
        st.number_input("Age (years)", min_value=0, max_value=200, value=None, step=1, key="in_age")
        st.selectbox("Sex", ["Not recorded", "F", "M", "O"], key="in_sex")
        st.number_input("Systolic BP (mmHg)", min_value=0, max_value=500, value=None, step=1, key="in_sbp",
                        help="Plausible range 60-260")
        st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=500, value=None, step=1, key="in_dbp")
    with b:
        st.number_input("Glucose (mg/dL)", min_value=0, max_value=2000, value=None, step=1, key="in_glu")
        st.selectbox("Glucose type", ["random", "fasting"], key="in_glu_type")
        st.number_input("BMI (kg/m²)", min_value=0.0, max_value=200.0, value=None, step=0.5, key="in_bmi")
        st.selectbox("Smoker", TRI, key="in_smoker")
        st.selectbox("Close relative with diabetes / BP / heart disease", TRI, key="in_family")
    with c:
        st.markdown("**Symptoms**")
        for s in SY_KEYS:
            st.selectbox(tc.SYMPTOM_LABEL[s], TRI, key="sy_" + s)
        if "ecg" in caps:
            st.selectbox("ECG result (available here)", ["Not done", "Normal", "Abnormal"], key="in_ecg")
        if "lab" in caps:
            st.number_input("HbA1c % (lab available here)", min_value=0.0, max_value=30.0, value=None, step=0.1,
                            key="in_hba1c")
    if ss.get("loaded"):
        st.caption("Loaded scenario: " + ss["loaded"])
    if st.button(tr("run"), type="primary"):
        f = collect_findings()
        res = triage(f)
        ss["res"], ss["findings"] = res, f
        ss["aid"] = STORE.save_assessment(origin, f, res) if res.get("status") == "ok" else None
        ss["origin_used"] = origin
    if "res" in ss:
        st.divider()
        show_result(ss["res"])

# --------------------------------------------------------------- 2 What-if
with tabs[1]:
    st.subheader("What-if: change a finding and watch the answer move")
    if "res" not in ss or ss["res"].get("status") != "ok" or ss["res"]["mode"] != "model":
        st.info("What-if analysis is available for completed model-based assessments without an urgent red-flag override. Follow urgent referral instructions first.")
    else:
        base_f, aid = ss["findings"], ss.get("aid") or "x"
        w1, w2, w3 = st.columns(3)
        st.caption("Measurements not present in the original assessment stay missing unless you explicitly enter a hypothetical value here.")
        sbp_default = int(base_f["sbp_mmhg"]) if base_f.get("sbp_mmhg") is not None else None
        glu_default = int(base_f["glucose_mg_dl"]) if base_f.get("glucose_mg_dl") is not None else None
        sbp = w1.number_input("Systolic BP (mmHg)", min_value=60, max_value=260, value=sbp_default,
                              step=1, key=f"wi_sbp_{aid}", help="Leave blank to keep missing; entering a value creates a what-if measurement.")
        glu = w2.number_input("Glucose (mg/dL)", min_value=20, max_value=700, value=glu_default,
                              step=1, key=f"wi_glu_{aid}", help="Leave blank to keep missing; entering a value creates a what-if measurement.")
        cols = w3
        sy_new = {}
        for s in SY_KEYS:
            cur = base_f.get("symptoms", {}).get(s)
            idx = 0 if cur is None else (2 if cur else 1)
            sy_new[s] = TRI_MAP[cols.selectbox(tc.SYMPTOM_LABEL[s], TRI, index=idx, key=f"wi_{s}_{aid}")]
        f2 = dict(base_f)
        # Missing measurements remain missing unless explicitly entered in this what-if view.
        if sbp is None:
            f2.pop("sbp_mmhg", None)
        else:
            f2["sbp_mmhg"] = float(sbp)
        if glu is None:
            f2.pop("glucose_mg_dl", None)
            f2.pop("glucose_type", None)
        else:
            f2["glucose_mg_dl"] = float(glu)
            f2.setdefault("glucose_type", "random")
        f2["symptoms"] = sy_new
        memo = ss.setdefault("memo", {})
        key = json.dumps([f2, m_w, u_w, origin, kill], sort_keys=True, default=str)
        if key not in memo:  # small session-local cache; bound memory in long-running demos
            if len(memo) >= 100:
                memo.clear()
            memo[key] = triage(f2)
        res2 = memo[key]
        l, r = st.columns(2)
        with l:
            st.markdown("**Before**")
            st.pyplot(prob_fig(ss["res"]))
            st.write(f"{ss['res']['risk_icon']} {tc.ACTION_LABEL[ss['res']['recommended_action']]}")
        with r:
            st.markdown("**After**")
            if res2.get("status") == "ok" and res2.get("probabilities"):
                st.pyplot(prob_fig(res2))
                st.write(f"{res2['risk_icon']} {tc.ACTION_LABEL[res2['recommended_action']]}   "
                         f"(refreshed in {res2['latency_ms']:.0f} ms; target under 500 ms)")
            else:
                st.error("; ".join(res2.get("errors", ["Could not compute"])))
        plt.close("all")
        st.caption("What-if results are computed on this screen only and are not saved.")

# --------------------------------------------------------------- 3 Why
with tabs[2]:
    res = ss.get("res")
    if not res or res.get("status") != "ok" or res["mode"] != "model":
        st.info("Explanations are available for completed model-based assessments. In an urgent red-flag case, follow referral instructions first.")
    else:
        st.subheader("Which findings moved the answer (leave-one-out)")
        if res["explanation"]:
            fig, ax = plt.subplots(figsize=(6, 2.2))
            lab = [f"{e['finding']} → {e['condition_label']}" for e in res["explanation"]]
            val = [e["delta"] for e in res["explanation"]]
            ax.barh(lab, val, color=["#c0392b" if v > 0 else "#2980b9" for v in val])
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("change in probability when this finding is removed (positive = finding raised it)")
            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close("all")
            for e in res["explanation"]:
                st.write(f"• **{e['finding']}** {e['direction']} the probability of {e['condition_label'].lower()} "
                         f"by {abs(e['delta'])*100:.0f} percentage points.")
        st.subheader("Expected utility of each action")
        eu = pd.DataFrame({"Action": [tc.ACTION_LABEL[a] for a in tc.ACTIONS],
                           "Expected utility (higher is better)": [round(res["action_eu"][a], 2) for a in tc.ACTIONS]})
        st.dataframe(eu, hide_index=True)
        st.caption("The utility table is a stated value judgement, to be agreed with a clinician.")
        st.subheader("How the answer depends on the missed-case weight")
        st.dataframe(pd.DataFrame(res["sensitivity"]).rename(columns={"missed_case_weight": "Missed-case weight",
                                                                      "action": "Chosen action"}), hide_index=True)
        st.subheader("Tests available at this facility, ranked by value of information per cost")
        tt = pd.DataFrame(res["tests"])
        if len(tt):
            tt = tt[["label", "gross_voi", "cost", "voi_per_cost", "net_voi"]].round(3)
            tt.columns = ["Test", "Value of information", "Cost", "Value per cost", "Net value"]
            st.dataframe(tt, hide_index=True)
        else:
            st.write("No further tests are available here or all findings are already entered.")
        st.subheader("Pipeline timing (ms)")
        st.write({k: round(v, 2) for k, v in res["step_ms"].items()})

# --------------------------------------------------------------- 4 Referral
with tabs[3]:
    st.subheader("Where to send the patient")
    st.markdown("**Facility status** (role: Medical Officer / facility admin in the pilot design). "
                "Changes are logged with a timestamp.")
    s1, s2 = st.columns(2)
    allf = list(NAME)
    new_closed = s1.multiselect("Closed facilities", allf, default=closed, format_func=lambda i: NAME[i])
    ecg_f = [i for i in allf if "ecg" in GRAPH.nodes[i]["base_caps"]]
    cur_noecg = [i for i, v in no_caps.items() if "ecg" in v]
    new_noecg = s2.multiselect("ECG not working at", ecg_f, default=[i for i in cur_noecg if i in ecg_f],
                               format_func=lambda i: NAME[i])
    if st.button("Save facility status"):
        STORE.set_facility_status(new_closed, {i: {"ecg"} for i in new_noecg})
        st.rerun()
    if status_ts:
        st.caption("Status last updated " + datetime.fromtimestamp(status_ts).strftime("%Y-%m-%d %H:%M:%S"))
    res = ss.get("res")
    default_act = res["recommended_action"] if res and res.get("status") == "ok" and \
        res["recommended_action"] in ("refer", "refer_urgently") else "refer"
    act = st.radio("Route for action", ["refer", "refer_urgently"], index=["refer", "refer_urgently"].index(default_act),
                   horizontal=True, format_func=lambda a: tc.ACTION_LABEL[a])
    p_high = res["state_probs"]["high"] if res and res.get("state_probs") else 0.0
    plan = tc.plan_referral(GRAPH, origin, act, p_high)
    if not plan["ok"]:
        st.error(plan["message"])
    else:
        if plan["note"]:
            st.warning(plan["note"])
        st.success(f"Go to **{plan['destination_name']}**: {plan['minutes']:.0f} min, {plan['km']:.0f} km. "
                   f"Needs: {', '.join(plan['required'])}.")
        rows = []
        for nm, r in (("A*", plan["astar"]), ("Uniform-cost", plan["ucs"]), ("Nearest by straight line", plan["nearest"])):
            if r:
                rows.append({"Method": nm, "Destination": NAME[r["destination"]], "Minutes": round(r["minutes"], 1),
                             "Nodes expanded": r["expanded"],
                             "Has needed capability and open": "yes" if nm != "Nearest by straight line" else
                             ("yes" if r["has_capability"] else "NO")})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.pyplot(map_fig(plan, origin))
    plt.close("all")
    if res and res.get("status") == "ok" and plan["ok"] and ss.get("aid"):
        st.download_button("Download referral slip (PDF)", slip_pdf(res, ss["aid"], origin, plan),
                           file_name=f"referral_slip_{ss['aid']}.pdf", mime="application/pdf")

# --------------------------------------------------------------- 5 Decision
with tabs[4]:
    st.subheader("Human decision: accept or override")
    res, aid = ss.get("res"), ss.get("aid")
    if not res or not aid:
        st.info("Run triage first.")
    else:
        st.write(f"Assessment `{aid}` recommends: **{tc.ACTION_LABEL[res['recommended_action']]}**")
        choice = st.radio("Your decision", ["Accept the recommendation", "Override"], horizontal=True)
        reasons = ["Clinical judgement: patient looks better than the tool suggests",
                   "Clinical judgement: patient looks worse than the tool suggests",
                   "Patient or family declines referral", "Transport or distance not feasible",
                   "Destination closed or unavailable", "Other (note on the paper register)"]
        reason = st.selectbox("Reason (required for override; no free text so no personal data)", reasons) \
            if choice == "Override" else None
        already_decided = STORE.has_decision(aid)
        if already_decided:
            st.info("A decision is already recorded for this assessment. Run a new assessment to record a new decision.")
        if st.button("Save decision", disabled=already_decided):
            try:
                needs = STORE.add_decision(aid, choice.startswith("Accept"), reason, bool(res["red_flags"]),
                                           res["recommended_action"])
                st.success("Decision saved.")
                if needs:
                    st.warning("Overriding an urgent result: flagged for Medical Officer review in the audit log.")
            except ValueError as exc:
                st.error(str(exc))
    st.subheader("Saved assessments")
    st.dataframe(STORE.assessments_df(), hide_index=True)
    ok, n = STORE.verify_chain()
    st.write(("✅ Audit chain verified" if ok else "❌ AUDIT CHAIN BROKEN") + f" ({n} entries). "
             "The hash chain detects inconsistent entries but is not tamper-proof storage; it stores IDs only, never clinical values.")
    st.dataframe(STORE.audit_df().head(30), hide_index=True)

# --------------------------------------------------------------- 6 Evaluation
with tabs[5]:
    st.subheader("Evaluation (run in the notebook; synthetic data)")
    if not os.path.exists("eval/eval_results.json"):
        st.info("Run the evaluation cells in the notebook first.")
    else:
        E = json.load(open("eval/eval_results.json"))
        st.caption(E["caveat"])
        st.markdown("**Model quality on a held-out test split** (criteria fixed before the test split was opened)")
        st.dataframe(pd.DataFrame(E["model_table"]), hide_index=True)
        if os.path.exists("eval/calibration.png"):
            st.image("eval/calibration.png", caption="Calibration (reliability) plot")
        st.markdown("**Decision quality: mean utility, closer to 0 is better** (scored against the true risk state, same utility table)")
        st.dataframe(pd.DataFrame(E["decision_table"]), hide_index=True)
        st.markdown("**Routing benchmark** " + f"({E['routing']['nodes']} synthetic facilities, {E['routing']['queries']} queries)")
        st.write({k: (round(v, 3) if isinstance(v, float) else v) for k, v in E["routing"].items()})
        st.markdown("**Latency (local engine)**")
        st.write(E["latency"])
        st.markdown("**Property tests**")
        st.dataframe(pd.DataFrame(E["property_tests"]), hide_index=True)

# --------------------------------------------------------------- 7 Scenarios
with tabs[6]:
    st.subheader("Preloaded patients for a repeatable demo")
    for name in SCENARIOS:
        c1, c2 = st.columns([4, 1])
        c1.write(name)
        c2.button("Load", key="load_" + name[:1], on_click=load_scenario, args=(name,))
    if ss.get("loaded"):
        st.success("Loaded. Open the **1 Assess** tab and press the button.")

# --------------------------------------------------------------- 8 About
with tabs[7]:
    st.subheader("About and limits")
    st.markdown("""
**Intended use (course prototype):** show how a Bayesian network, expected utility, value of information and A* search can
support a health worker's referral decision at levels 1-2, one condition cluster (hypertension, diabetes, heart-disease risk).

**Not intended for:** real patients, diagnosis, or any claim of clinical accuracy. Probabilities come from a **synthetic**
cohort built from a stated generating process. The red-flag rules and the utility table are illustrative and need
clinician review. This prototype claims no CDSCO, ICMR or DPDP approval.

**Human in charge:** the system recommends, explains and records. The worker accepts or overrides, with a reason.
""")
    meta = MODEL.meta
    st.markdown("**Model card**")
    st.write({"version": tc.MODEL_VERSION, "sha256": MODEL.sha256[:16] + "...", "training rows": meta.get("n_train"),
              "data": meta.get("data"), "equivalent sample size": meta.get("ess"), "samples K": MODEL.K})
    st.markdown("**Hazard log (short)**")
    st.dataframe(pd.DataFrame([
        ["Missed high-risk patient", "Weak model, sparse inputs", "Red-flag rules first; conservative weights; 'treat locally' blocked on thin evidence; human decides", "Medium"],
        ["Blind trust in output", "Over-reliance", "Intervals, confidence, reasons and limits on every screen; override logged", "Medium"],
        ["Wrong facility status", "Stale or false 'open'", "Role-limited changes, timestamp shown, audit log", "Medium"],
        ["Tampered model", "Edited file", "SHA-256 check at load; refuses to start otherwise; kill switch", "Low"],
        ["Privacy slip", "Real data typed in", "No name/phone/ID fields; pseudonymous IDs; logs hold no findings", "Low"],
    ], columns=["Hazard", "Cause", "Control", "Residual risk"]), hide_index=True)
    st.markdown("**Tier 2 (pilot) design, not built here:** React PWA, FastAPI, PostgreSQL + PostGIS, Redis, OIDC login with OTP + PIN, "
                "row-level security, CI/CD, Prometheus/Grafana monitoring, AWS Mumbai hosting. See the design plan.")
st.caption("AAROGYA MARG · " + tc.DISCLAIMER)