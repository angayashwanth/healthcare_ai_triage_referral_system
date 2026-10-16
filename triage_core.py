"""AAROGYA MARG - triage_core
Pure decision engine: no web, database or UI imports.
Pipeline: validate -> red flags -> Bayesian network -> confidence -> expected utility
          -> value of information -> routing (A*) -> explanation.
PROTOTYPE FOR COURSEWORK. NOT A MEDICAL DEVICE. A CLINICIAN DECIDES.
"""
import hashlib
import hmac
import heapq
import itertools
import json
import logging
import math
import time

import numpy as np

MODEL_VERSION = "bn-2026.10.1"
DISCLAIMER = "Prototype for coursework. Not a medical device. A clinician decides."

# ----------------------------------------------------------------------------
# 1. Bayesian network definition (expert-fixed structure, learned tables)
# ----------------------------------------------------------------------------
STATES = {
    "age": ["<40", "40-59", "60+"],
    "bmi": ["<25", "25-29.9", "30+"],
    "smoker": ["no", "yes"],
    "family": ["no", "yes"],
    "htn": ["no", "yes"],
    "dm": ["no", "yes"],
    "hr": ["no", "yes"],
    "sbp": ["<120", "120-139", "140-159", "160+"],
    "glucose": ["normal", "raised", "high"],
    "headache": ["no", "yes"],
    "polyuria": ["no", "yes"],
    "tiredness": ["no", "yes"],
    "chest": ["no", "yes"],
    "breathless": ["no", "yes"],
    "ecg": ["normal", "abnormal"],
    "hba1c": ["<6.5", "6.5+"],
}
PARENTS = {
    "age": (), "bmi": (), "smoker": (), "family": (),
    "htn": ("age", "bmi"),
    "dm": ("age", "bmi", "family"),
    "hr": ("age", "smoker", "family", "htn", "dm"),
    "sbp": ("htn",), "headache": ("htn",),
    "glucose": ("dm",), "polyuria": ("dm",), "tiredness": ("dm",), "hba1c": ("dm",),
    "chest": ("hr",), "breathless": ("hr",), "ecg": ("hr",),
}
ORDER = list(PARENTS.keys())
TARGETS = ("htn", "dm", "hr")
CARD = {v: len(s) for v, s in STATES.items()}
LETTER = {v: chr(ord("a") + i) for i, v in enumerate(ORDER)}
COND_LABEL = {"htn": "Hypertension", "dm": "Diabetes", "hr": "Heart-disease risk"}
SYMPTOMS = {  # finding key -> network variable
    "headache": "headache", "polyuria": "polyuria", "tiredness": "tiredness",
    "chest_discomfort": "chest", "breathlessness": "breathless",
}
SYMPTOM_LABEL = {
    "headache": "Headache", "polyuria": "Excess thirst / urination", "tiredness": "Tiredness",
    "chest_discomfort": "Chest discomfort", "breathlessness": "Breathlessness",
}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _cat(rng, table, parent):
    """Sample a categorical child. table: (parent_card, k) probabilities."""
    table = np.asarray(table)
    cum = table[parent].cumsum(axis=1)
    u = rng.random(len(parent))[:, None]
    return np.minimum((u > cum).sum(axis=1), table.shape[1] - 1)


def gen_cohort(n, seed=7):
    """SYNTHETIC cohort from a stated clinical-style generating process.
    It stands in for a public dataset; results only show the method works on
    this process, not that it works in a village."""
    rng = np.random.default_rng(seed)
    d = {}
    d["age"] = rng.choice(3, n, p=[0.40, 0.35, 0.25])
    d["bmi"] = rng.choice(3, n, p=[0.40, 0.35, 0.25])
    d["smoker"] = (rng.random(n) < 0.20).astype(int)
    d["family"] = (rng.random(n) < 0.25).astype(int)
    d["htn"] = (rng.random(n) < _sig(-2.3 + 0.9 * d["age"] + 0.55 * d["bmi"])).astype(int)
    d["dm"] = (rng.random(n) < _sig(-3.0 + 0.55 * d["age"] + 0.65 * d["bmi"] + 0.9 * d["family"])).astype(int)
    d["hr"] = (rng.random(n) < _sig(-4.0 + 0.8 * d["age"] + 1.0 * d["smoker"] + 0.7 * d["family"]
                                    + 1.1 * d["htn"] + 0.9 * d["dm"])).astype(int)
    d["sbp"] = _cat(rng, [[.45, .38, .14, .03], [.03, .12, .40, .45]], d["htn"])
    d["glucose"] = _cat(rng, [[.80, .15, .05], [.10, .30, .60]], d["dm"])
    d["headache"] = _cat(rng, [[.88, .12], [.60, .40]], d["htn"])
    d["polyuria"] = _cat(rng, [[.94, .06], [.55, .45]], d["dm"])
    d["tiredness"] = _cat(rng, [[.80, .20], [.50, .50]], d["dm"])
    d["hba1c"] = _cat(rng, [[.96, .04], [.15, .85]], d["dm"])
    d["chest"] = _cat(rng, [[.95, .05], [.55, .45]], d["hr"])
    d["breathless"] = _cat(rng, [[.92, .08], [.60, .40]], d["hr"])
    d["ecg"] = _cat(rng, [[.92, .08], [.40, .60]], d["hr"])
    import pandas as pd
    return pd.DataFrame({v: d[v] for v in ORDER})


def true_risk_state(df):
    """0 low, 1 moderate, 2 high (same definition the engine uses)."""
    hr = df["hr"].values
    mod = ((df["htn"].values == 1) | (df["dm"].values == 1)) & (hr == 0)
    return np.where(hr == 1, 2, np.where(mod, 1, 0))


# ----------------------------------------------------------------------------
# 2. Model: Dirichlet posterior tables, JSON artifact with SHA-256
# ----------------------------------------------------------------------------
class Model:
    def __init__(self, alphas, K=200, seed=11, meta=None):
        self.alphas = {v: np.asarray(a, float) for v, a in alphas.items()}
        self.K, self.seed, self.meta = K, seed, meta or {}
        rng = np.random.default_rng(seed)
        self.samples, self.mean = {}, {}
        for v in ORDER:
            a = self.alphas[v]
            g = rng.gamma(a, size=(K,) + a.shape)
            self.samples[v] = g / g.sum(axis=-1, keepdims=True)
            self.mean[v] = (a / a.sum(axis=-1, keepdims=True))[None]

    def cpts(self, use_mean):
        return self.mean if use_mean else self.samples


def train_model(df, ess=8.0, K=200, seed=11, meta=None):
    """Fit CPTs after validating the categorical training data and parameters."""
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    missing = [v for v in ORDER if v not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Training data must contain at least one row")
    if not np.isfinite(ess) or ess <= 0:
        raise ValueError("ess must be a finite positive number")
    if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError("K must be a positive integer")
    data = df[ORDER]
    for v in ORDER:
        vals = data[v].to_numpy()
        if not np.all(np.isfinite(vals)) or not np.all(vals == np.floor(vals)):
            raise ValueError(f"Training column {v!r} must contain integer state indices")
        if np.any(vals < 0) or np.any(vals >= CARD[v]):
            raise ValueError(f"Training column {v!r} contains a state outside 0..{CARD[v]-1}")
    alphas = {}
    for v, pa in PARENTS.items():
        shape = [CARD[p] for p in pa] + [CARD[v]]
        counts = np.zeros(shape)
        idx = tuple(data[col].to_numpy(dtype=int) for col in pa) + (data[v].to_numpy(dtype=int),)
        np.add.at(counts, idx, 1)
        q = int(np.prod([CARD[parent] for parent in pa])) if pa else 1
        alphas[v] = counts + ess / (q * CARD[v])
    return Model(alphas, K=int(K), seed=int(seed), meta=meta)


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def save_model(model, path):
    body = {
        "version": MODEL_VERSION, "K": model.K, "seed": model.seed, "meta": model.meta,
        "alphas": {v: model.alphas[v].tolist() for v in ORDER},
    }
    digest = hashlib.sha256(_canon(body).encode()).hexdigest()
    with open(path, "w") as f:
        json.dump({"body": body, "sha256": digest}, f)
    return digest


def load_model(path):
    """Load and validate a versioned model artifact; checksum is integrity, not authenticity."""
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    if not isinstance(blob, dict) or not isinstance(blob.get("body"), dict) or not isinstance(blob.get("sha256"), str):
        raise ValueError("Malformed model artifact: expected body and sha256 fields")
    body = blob["body"]
    digest = hashlib.sha256(_canon(body).encode()).hexdigest()
    if not hmac.compare_digest(digest, blob["sha256"]):
        raise ValueError("Model file failed SHA-256 check - refusing to load.")
    if body.get("version") != MODEL_VERSION:
        raise ValueError(f"Unsupported model version {body.get('version')!r}; expected {MODEL_VERSION!r}")
    K, seed, alphas, meta = body.get("K"), body.get("seed"), body.get("alphas"), body.get("meta")
    if isinstance(K, bool) or not isinstance(K, int) or not 1 <= K <= 10000:
        raise ValueError("Invalid model K: expected an integer from 1 to 10000")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Invalid model seed")
    if not isinstance(alphas, dict) or set(alphas) != set(ORDER):
        raise ValueError("Model CPT keys do not match the expected network variables")
    for var in ORDER:
        expected = tuple(CARD[x] for x in PARENTS[var]) + (CARD[var],)
        arr = np.asarray(alphas[var], dtype=float)
        if arr.shape != expected:
            raise ValueError(f"Invalid CPT shape for {var}: expected {expected}, got {arr.shape}")
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError(f"Invalid Dirichlet parameters for {var}: values must be finite and positive")
    if meta is not None and not isinstance(meta, dict):
        raise ValueError("Model metadata must be a JSON object")
    m = Model(alphas, K=K, seed=seed, meta=meta)
    m.sha256 = digest
    return m


# ----------------------------------------------------------------------------
# 3. Exact inference (variable elimination via einsum), batched over K samples
# ----------------------------------------------------------------------------
def posterior(model, ev, use_mean=False):
    """Return posterior and evidence likelihood for validated discrete evidence."""
    if not isinstance(ev, dict):
        raise TypeError("Evidence must be a dictionary")
    for var, state in ev.items():
        if var not in CARD or isinstance(state, bool) or not isinstance(state, (int, np.integer)) or not 0 <= state < CARD[var]:
            raise ValueError(f"Invalid evidence state for {var!r}: {state!r}")
    cpts = model.cpts(use_mean)
    ops, subs = [], []
    for v in ORDER:
        if v not in TARGETS and v in PARENTS and not PARENTS[v] == () and v not in ev:
            continue  # barren leaf (not observed)
        axvars = list(PARENTS[v]) + [v]
        idx = (slice(None),) + tuple(ev.get(x, slice(None)) for x in axvars)
        arr = cpts[v][idx]
        rem = [x for x in axvars if x not in ev]
        ops.append(arr)
        subs.append("z" + "".join(LETTER[x] for x in rem))
    out = "z" + "".join(LETTER[t] for t in TARGETS)
    joint = np.einsum(",".join(subs) + "->" + out, *ops, optimize=False)
    Z = joint.sum(axis=(1, 2, 3))
    if not np.all(np.isfinite(Z)) or np.any(Z <= 0):
        raise FloatingPointError("Bayesian inference produced a non-finite or zero evidence likelihood")
    post = joint / Z[:, None, None, None]
    if not np.all(np.isfinite(post)):
        raise FloatingPointError("Bayesian inference produced non-finite posterior probabilities")
    return post, Z, joint


def cond_probs(post):
    return {
        "htn": post[:, 1, :, :].sum(axis=(1, 2)),
        "dm": post[:, :, 1, :].sum(axis=(1, 2)),
        "hr": post[:, :, :, 1].sum(axis=(1, 2)),
    }


def state_probs(post):
    """P(low, moderate, high) per sample. high = heart risk; moderate = HTN or DM without it."""
    high = post[:, :, :, 1].sum(axis=(1, 2))
    mod = post[:, 1, 0, 0] + post[:, 0, 1, 0] + post[:, 1, 1, 0]
    low = post[:, 0, 0, 0]
    return np.stack([low, mod, high], axis=1)


def summarize(p):
    return {
        "median": float(np.median(p)),
        "low80": float(np.percentile(p, 10)),
        "high80": float(np.percentile(p, 90)),
    }


# ----------------------------------------------------------------------------
# 4. Validation and encoding of findings (blank = None, never 0)
# ----------------------------------------------------------------------------
RANGES = {
    "age": (18, 110), "sbp_mmhg": (60, 260), "dbp_mmhg": (30, 160),
    "glucose_mg_dl": (20, 700), "bmi_kg_m2": (10, 70), "hba1c_pct": (3, 18),
}


def _parse_bool(value, field, errors):
    """Parse explicit boolean forms; never use Python truthiness for user input."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("yes", "true", "1"):
            return True
        if token in ("no", "false", "0"):
            return False
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value) and value in (0, 1):
        return bool(value)
    errors.append(f"{field}: expected yes/no, true/false, 1/0, or a boolean")
    return None


def validate(findings):
    errors, warnings, clean = [], [], {}
    if findings is None:
        f = {}
    elif isinstance(findings, dict):
        f = findings
    else:
        return {"errors": ["findings must be a dictionary"], "warnings": [], "clean": {}}
    for k, (lo, hi) in RANGES.items():
        v = f.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        try:
            v = float(v)
        except (TypeError, ValueError, OverflowError):
            errors.append(f"{k}: not a number")
            continue
        if not np.isfinite(v) or not (lo <= v <= hi):
            errors.append(f"{k}={v:g} is outside the plausible finite range {lo}-{hi}. Please re-check; "
                          f"the value was NOT silently changed.")
        else:
            clean[k] = v
    if "sbp_mmhg" in clean and "dbp_mmhg" in clean and clean["dbp_mmhg"] >= clean["sbp_mmhg"]:
        errors.append("Diastolic BP must be lower than systolic BP; re-check the paired reading.")
        clean.pop("sbp_mmhg", None)
        clean.pop("dbp_mmhg", None)
    if "glucose_mg_dl" in clean:
        gt = f.get("glucose_type") or "random"
        if f.get("glucose_type") not in ("random", "fasting"):
            warnings.append("Glucose type not given or invalid - treated as random.")
        clean["glucose_type"] = gt if gt in ("random", "fasting") else "random"
    sex = f.get("sex")
    clean["sex"] = sex if sex in ("F", "M", "O") else None
    for k in ("smoker", "family_history"):
        clean[k] = _parse_bool(f.get(k), k, errors)
    sy = f.get("symptoms") or {}
    if isinstance(sy, (list, tuple, set)):
        sy = {s: True for s in sy}
    if not isinstance(sy, dict):
        errors.append("symptoms must be a dictionary or a list of symptom names")
        sy = {}
    clean_sy = {}
    for k in SYMPTOMS:
        if k in sy:
            clean_sy[k] = _parse_bool(sy[k], f"symptoms.{k}", errors)
    clean["symptoms"] = clean_sy
    e = f.get("ecg")
    if e not in (None, "", "normal", "abnormal"):
        errors.append("ecg: expected 'normal', 'abnormal', or missing")
    clean["ecg"] = e if e in ("normal", "abnormal") else None
    return {"errors": errors, "warnings": warnings, "clean": clean}


def _band(x, cuts):
    return int(sum(x >= c for c in cuts))


def encode(clean):
    """findings -> evidence dict {network variable: state index}."""
    ev = {}
    if "age" in clean:
        ev["age"] = _band(clean["age"], (40, 60))
    if "sbp_mmhg" in clean:
        ev["sbp"] = _band(clean["sbp_mmhg"], (120, 140, 160))
    if "glucose_mg_dl" in clean:
        cuts = (100, 126) if clean.get("glucose_type") == "fasting" else (140, 200)
        ev["glucose"] = _band(clean["glucose_mg_dl"], cuts)
    if "bmi_kg_m2" in clean:
        ev["bmi"] = _band(clean["bmi_kg_m2"], (25, 30))
    if clean.get("smoker") is not None:
        ev["smoker"] = int(clean["smoker"])
    if clean.get("family_history") is not None:
        ev["family"] = int(clean["family_history"])
    for k, var in SYMPTOMS.items():
        if clean.get("symptoms", {}).get(k) is not None:
            ev[var] = int(clean["symptoms"][k])
    if clean.get("ecg") is not None:
        ev["ecg"] = 0 if clean["ecg"] == "normal" else 1
    if "hba1c_pct" in clean:
        ev["hba1c"] = int(clean["hba1c_pct"] >= 6.5)
    return ev


def describe_finding(var, clean):
    """Plain-language label of an observed finding for explanations."""
    s = clean
    if var == "sbp":
        return f"Systolic BP {s['sbp_mmhg']:g} mmHg"
    if var == "glucose":
        return f"{(s.get('glucose_type') or 'random').capitalize()} glucose {s['glucose_mg_dl']:g} mg/dL"
    if var == "age":
        return f"Age {s['age']:g}"
    if var == "bmi":
        return f"BMI {s['bmi_kg_m2']:g}"
    if var == "smoker":
        return "Smoker" if s["smoker"] else "Non-smoker"
    if var == "family":
        return "Family history present" if s["family_history"] else "No family history"
    if var == "ecg":
        return f"ECG {s['ecg']}"
    if var == "hba1c":
        return f"HbA1c {s['hba1c_pct']:g}%"
    inv = {v: k for k, v in SYMPTOMS.items()}
    if var in inv:
        yes = s["symptoms"][inv[var]]
        return SYMPTOM_LABEL[inv[var]] + (" present" if yes else " absent")
    return var


# ----------------------------------------------------------------------------
# 5. Red-flag rules (run BEFORE the model; the model cannot outvote them)
# ----------------------------------------------------------------------------
def red_flags(clean):
    """Illustrative guideline-style rules. To be reviewed by a clinician (see hazard log)."""
    flags = []
    sbp, dbp, glu = clean.get("sbp_mmhg"), clean.get("dbp_mmhg"), clean.get("glucose_mg_dl")
    sy = clean.get("symptoms", {})
    if (sbp is not None and sbp >= 180) or (dbp is not None and dbp >= 120):
        flags.append({"code": "RF1", "message": "Severe hypertension (SBP 180+ or DBP 120+)"})
    if sy.get("chest_discomfort") and sy.get("breathlessness"):
        flags.append({"code": "RF2", "message": "Chest discomfort with breathlessness"})
    if glu is not None and glu >= 300 and (sy.get("polyuria") or sy.get("tiredness")):
        flags.append({"code": "RF3", "message": "Very high glucose (300+) with symptoms"})
    if glu is not None and glu < 54:
        flags.append({"code": "RF4", "message": "Very low glucose (below 54)"})
    if sbp is not None and sbp < 90 and (sy.get("chest_discomfort") or sy.get("breathlessness")):
        flags.append({"code": "RF5", "message": "Low BP (SBP below 90) with chest symptoms"})
    return flags


def guideline_checklist(clean, flags):
    """Kill-switch fallback: plain thresholds, no model."""
    if flags:
        return "refer_urgently", "Guideline checklist (model off): red flag present."
    sbp, glu = clean.get("sbp_mmhg"), clean.get("glucose_mg_dl")
    gt = clean.get("glucose_type", "random")
    gcut = 126 if gt == "fasting" else 200
    sy = clean.get("symptoms", {})
    if (sbp is not None and sbp >= 140) or (glu is not None and glu >= gcut) or sy.get("chest_discomfort"):
        return "refer", "Guideline checklist (model off): BP, glucose or chest symptom above threshold."
    return "treat_locally", "Guideline checklist (model off): no threshold crossed; review at next visit."


# ----------------------------------------------------------------------------
# 6. Expected utility and value of information
# ----------------------------------------------------------------------------
ACTIONS = ["treat_locally", "one_more_test", "refer", "refer_urgently"]
ACTION_LABEL = {
    "treat_locally": "Treat locally", "one_more_test": "Do one more test",
    "refer": "Refer", "refer_urgently": "Refer urgently",
}
RISK_STATES = ["low", "moderate", "high"]
BASE_U = np.array([[0, -20, -100], [-2, -2, -10], [-10, -3, -5], [-25, -15, 0]], float)


def utility_matrix(missed_case_weight=10.0, unneeded_referral_weight=1.0):
    """Illustrative table (to be agreed with a clinician). The missed-case weight scales
    under-triage cells; the unneeded-referral weight scales over-triage cells."""
    U = BASE_U.copy()
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        U[i, j] *= missed_case_weight / 10.0
    for i, j in [(2, 0), (3, 0), (3, 1)]:
        U[i, j] *= unneeded_referral_weight
    return U


def expected_utility(P, U):
    return U @ np.asarray(P)  # shape (4,)


def choose_action(P, U, allow=ACTIONS):
    eu = expected_utility(P, U)
    best, best_v = None, -1e18
    for i, a in enumerate(ACTIONS):
        if a in allow and eu[i] > best_v + 1e-12:
            best, best_v = a, eu[i]
    return best, eu


def action_sensitivity(P, u_weight=1.0, ratios=(1, 2, 5, 10, 20, 50)):
    return [{"missed_case_weight": r, "action": choose_action(P, utility_matrix(r, u_weight))[0]} for r in ratios]


TESTS = {
    "measure_bp": {"label": "Measure blood pressure", "var": "sbp", "cost": 0.3, "needs": "bp_cuff"},
    "glucose_strip": {"label": "Finger-prick glucose", "var": "glucose", "cost": 0.5, "needs": "glucometer"},
    "measure_bmi": {"label": "Measure weight and height (BMI)", "var": "bmi", "cost": 0.3, "needs": "scale"},
    "ask_smoking": {"label": "Ask about smoking", "var": "smoker", "cost": 0.1, "needs": None},
    "ask_family": {"label": "Ask about family history", "var": "family", "cost": 0.1, "needs": None},
    "ask_headache": {"label": "Ask about headache", "var": "headache", "cost": 0.1, "needs": None},
    "ask_polyuria": {"label": "Ask about thirst / urination", "var": "polyuria", "cost": 0.1, "needs": None},
    "ask_tiredness": {"label": "Ask about tiredness", "var": "tiredness", "cost": 0.1, "needs": None},
    "ask_chest": {"label": "Ask about chest discomfort", "var": "chest", "cost": 0.1, "needs": None},
    "ask_breathless": {"label": "Ask about breathlessness", "var": "breathless", "cost": 0.1, "needs": None},
    "ecg_test": {"label": "12-lead ECG", "var": "ecg", "cost": 1.5, "needs": "ecg"},
    "hba1c_lab": {"label": "HbA1c blood test", "var": "hba1c", "cost": 1.0, "needs": "lab"},
}
FINAL_ACTIONS = ["treat_locally", "refer", "refer_urgently"]


def rank_tests(model, ev, U, caps, use_mean=False):
    """Rank tests the facility actually has by (gross VOI over the final decision) / cost.
    P(o, s | e) is averaged over the K table samples so the terms are consistent."""
    post, Z, _ = posterior(model, ev, use_mean)
    P0 = state_probs(post).mean(axis=0)
    rows = U[[ACTIONS.index(a) for a in FINAL_ACTIONS]]
    base = float((rows @ P0).max())
    out = []
    for name, t in TESTS.items():
        v = t["var"]
        if v in ev or (t["needs"] and t["needs"] not in caps):
            continue
        gross = 0.0
        for o in range(CARD[v]):
            e2 = dict(ev)
            e2[v] = o
            post2, Z2, _ = posterior(model, e2, use_mean)
            joint_os = ((Z2 / Z)[:, None] * state_probs(post2)).mean(axis=0)  # P(o, s | e)
            gross += float((rows @ joint_os).max())
        gross = max(gross - base, 0.0)
        out.append({"name": name, "label": t["label"], "gross_voi": gross, "cost": t["cost"],
                    "voi_per_cost": gross / t["cost"], "net_voi": gross - t["cost"]})
    out.sort(key=lambda r: r["voi_per_cost"], reverse=True)
    return out


# ----------------------------------------------------------------------------
# 7. Facility graph and routing (A*, uniform-cost, nearest-by-straight-line)
# ----------------------------------------------------------------------------
LEVEL = {"SC": 1, "PHC": 2, "CHC": 3, "DH": 4}
BASE_CAPS = {
    "SC": {"bp_cuff", "glucometer", "scale"},
    "PHC": {"bp_cuff", "glucometer", "scale", "physician", "lab"},
    "CHC": {"bp_cuff", "glucometer", "scale", "physician", "lab", "ecg", "xray"},
    "DH": {"bp_cuff", "glucometer", "scale", "physician", "lab", "ecg", "xray", "icu", "cardiologist"},
}
VMAX_KMH = 60.0  # faster than any edge speed -> heuristic never overestimates


class FacilityGraph:
    def __init__(self):
        self.nodes, self.adj = {}, {}

    def add_node(self, nid, name, ftype, x, y, caps=None, open_=True):
        self.nodes[nid] = {"id": nid, "name": name, "type": ftype, "level": LEVEL[ftype],
                           "x": float(x), "y": float(y),
                           "base_caps": set(caps if caps is not None else BASE_CAPS[ftype]),
                           "caps": set(caps if caps is not None else BASE_CAPS[ftype]), "open": open_}
        self.adj.setdefault(nid, [])

    def add_edge(self, a, b, km, minutes):
        self.adj[a].append((b, minutes, km))
        self.adj[b].append((a, minutes, km))

    def straight_km(self, a, b):
        A, B = self.nodes[a], self.nodes[b]
        return math.hypot(A["x"] - B["x"], A["y"] - B["y"])

    def apply_status(self, closed=(), no_caps=None):
        """closed: iterable of facility ids; no_caps: {facility_id: set(capabilities not working)}."""
        no_caps = no_caps or {}
        for nid, n in self.nodes.items():
            n["open"] = nid not in set(closed)
            n["caps"] = set(n["base_caps"]) - set(no_caps.get(nid, ()))
        return self


def _connect(g, rng, k=3):
    ids = list(g.nodes)
    seen = set()

    def link(a, b):
        key = tuple(sorted((a, b)))
        if key in seen:
            return
        seen.add(key)
        km = g.straight_km(a, b) * rng.uniform(1.1, 1.4)
        g.add_edge(a, b, km, km / rng.uniform(25, 45) * 60)

    for a in ids:
        near = sorted((b for b in ids if b != a), key=lambda b: g.straight_km(a, b))[:k]
        for b in near:
            link(a, b)
    # connect components
    while True:
        comp = _components(g)
        if len(comp) == 1:
            return
        best = None
        for a in comp[0]:
            for c in comp[1:]:
                for b in c:
                    d = g.straight_km(a, b)
                    if best is None or d < best[0]:
                        best = (d, a, b)
        link(best[1], best[2])


def _components(g):
    seen, comps = set(), []
    for s in g.nodes:
        if s in seen:
            continue
        stack, c = [s], []
        seen.add(s)
        while stack:
            n = stack.pop()
            c.append(n)
            for m, _, _ in g.adj[n]:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        comps.append(c)
    return comps


DEMO_FACILITIES = [
    ("sc-01", "SC Kondapur", "SC", 8, 10), ("sc-02", "SC Lingapur", "SC", 14, 32),
    ("sc-03", "SC Mallepalli", "SC", 30, 6), ("sc-04", "SC Devarapalli", "SC", 46, 14),
    ("sc-05", "SC Rampur", "SC", 52, 38), ("sc-06", "SC Gokulam", "SC", 22, 50),
    ("phc-01", "PHC Sirigiri", "PHC", 20, 18), ("phc-02", "PHC Veeranagar", "PHC", 38, 26),
    ("phc-03", "PHC Chintalapuram", "PHC", 36, 48),
    ("chc-01", "CHC Raghavapuram", "CHC", 28, 34), ("chc-02", "CHC Somapuram", "CHC", 50, 52),
    ("dh-01", "District Hospital Aarogyapur", "DH", 44, 40),
]


def make_demo_graph(seed=3):
    g = FacilityGraph()
    for nid, name, t, x, y in DEMO_FACILITIES:
        g.add_node(nid, name, t, x, y)
    # phc-02 and phc-01 differ in ECG availability for the demo
    g.nodes["phc-02"]["base_caps"].add("ecg")
    g.nodes["phc-02"]["caps"].add("ecg")
    _connect(g, np.random.default_rng(seed), k=3)
    return g


def make_large_graph(n=150, seed=5, area=120.0):
    rng = np.random.default_rng(seed)
    g = FacilityGraph()
    types = rng.choice(["SC", "PHC", "CHC", "DH"], n, p=[0.70, 0.20, 0.07, 0.03])
    for i in range(n):
        g.add_node(f"n{i:03d}", f"Facility {i}", str(types[i]), rng.uniform(0, area), rng.uniform(0, area),
                   open_=bool(rng.random() < 0.95))
    _connect(g, rng, k=4)
    return g


def required_caps(action, p_high=0.0):
    if action == "refer_urgently":
        return {"physician", "ecg"}
    if action == "refer":
        return {"physician", "ecg"} if p_high >= 0.5 else {"physician"}
    return set()


def _search(g, start, goals, use_heuristic):
    goals = set(goals)

    def h(n):
        if not use_heuristic:
            return 0.0
        return min(g.straight_km(n, t) for t in goals) / VMAX_KMH * 60.0

    pq = [(h(start), 0.0, start)]
    best, parent, closed, expanded = {start: 0.0}, {start: None}, set(), 0
    while pq:
        f, gc, n = heapq.heappop(pq)
        if n in closed:
            continue
        closed.add(n)
        expanded += 1
        if n in goals:
            path = []
            while n is not None:
                path.append(n)
                n = parent[n]
            path.reverse()
            km = sum(next(k for m, _, k in g.adj[a] if m == b) for a, b in zip(path, path[1:]))
            return {"path": path, "minutes": gc, "km": km, "expanded": expanded, "destination": path[-1]}
        for m, mins, _ in g.adj[n]:
            ng = gc + mins
            if ng < best.get(m, 1e18) - 1e-12:
                best[m] = ng
                parent[m] = n
                heapq.heappush(pq, (ng + h(m), ng, m))
    return None


def astar(g, start, goals):
    return _search(g, start, goals, True)


def ucs(g, start, goals):
    return _search(g, start, goals, False)


def nearest_straight_line(g, start):
    """Baseline: nearest open higher-level facility by straight line; may lack capability."""
    cands = [n for n, v in g.nodes.items() if v["open"] and v["level"] > g.nodes[start]["level"]]
    if not cands:
        return None
    t = min(cands, key=lambda n: g.straight_km(start, n))
    r = ucs(g, start, [t])
    return r


def plan_referral(g, origin, action, p_high=0.0):
    """Route to a capable open higher-level facility; fail clearly if origin is unavailable."""
    need = required_caps(action, p_high)
    if g is None or origin not in g.nodes:
        return {"ok": False, "message": "Origin facility is not configured in the routing graph. Contact the Medical Officer and arrange transport.",
                "required": sorted(need)}
    lvl = g.nodes[origin]["level"]
    note = None
    goals = []
    for req in [need, {"physician"} & need, set()]:
        goals = [n for n, v in g.nodes.items() if v["open"] and v["level"] > lvl and req <= v["caps"]]
        if goals:
            if req != need:
                note = ("No open facility has all required capabilities "
                        f"({', '.join(sorted(need))}); routing to the nearest open higher-level facility instead. "
                        "Phone ahead and arrange transport.")
            break
    if not goals:
        return {"ok": False, "message": "No open higher-level facility is reachable. Contact the Medical Officer "
                                        "and arrange emergency transport.", "required": sorted(need)}
    a, u, n = astar(g, origin, goals), ucs(g, origin, goals), nearest_straight_line(g, origin)
    dest = g.nodes[a["destination"]]
    return {"ok": True, "required": sorted(need), "note": note, "algorithm": "astar",
            "destination": dest["id"], "destination_name": dest["name"], "path": a["path"],
            "minutes": round(a["minutes"], 1), "km": round(a["km"], 1),
            "astar": a, "ucs": u,
            "nearest": None if n is None else {**n, "has_capability": need <= g.nodes[n["destination"]]["caps"]
                                               and g.nodes[n["destination"]]["open"]}}


# ----------------------------------------------------------------------------
# 8. Explanation (leave-one-out)
# ----------------------------------------------------------------------------
def explain(model, ev, clean, top=3):
    base_post, _, _ = posterior(model, ev, True)
    base = {c: float(p[0]) for c, p in cond_probs(base_post).items()}
    rows = []
    for var in list(ev):
        e2 = {k: v for k, v in ev.items() if k != var}
        p2 = {c: float(p[0]) for c, p in cond_probs(posterior(model, e2, True)[0]).items()}
        deltas = {c: base[c] - p2[c] for c in TARGETS}
        c_best = max(deltas, key=lambda c: abs(deltas[c]))
        rows.append({"var": var, "finding": describe_finding(var, clean), "condition": c_best,
                     "condition_label": COND_LABEL[c_best], "delta": deltas[c_best],
                     "direction": "raised" if deltas[c_best] > 0 else "lowered"})
    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return rows[:top]


# ----------------------------------------------------------------------------
# 9. Logging that never contains findings
# ----------------------------------------------------------------------------
log = logging.getLogger("aarogya")
log.addHandler(logging.NullHandler())  # silent unless the app attaches a handler


def _log(**kw):
    log.info(json.dumps(kw, sort_keys=True))


# ----------------------------------------------------------------------------
# 10. The full pipeline
# ----------------------------------------------------------------------------
RISK_UI = {
    "treat_locally": ("green", "LOW", "🟢"),
    "one_more_test": ("yellow", "NEEDS MORE INFORMATION", "🟡"),
    "refer": ("amber", "REFER", "🟠"),
    "refer_urgently": ("red", "URGENT", "🔴"),
}


def _confidence(widths, n_obs):
    w = max(widths)
    insufficient = (w > 0.30) or (n_obs < 3)
    if insufficient or w >= 0.25 or n_obs < 4:
        level = "low"
    elif w < 0.12 and n_obs >= 6:
        level = "high"
    else:
        level = "medium"
    return level, insufficient, w


def _summary(probs, action, referral, flags):
    likely = [COND_LABEL[c].lower() for c in TARGETS if probs[c]["median"] >= 0.5]
    possible = [COND_LABEL[c].lower() for c in TARGETS if 0.3 <= probs[c]["median"] < 0.5]
    parts = []
    if flags:
        parts.append("Red flag: " + "; ".join(f["message"] for f in flags) + ".")
    if likely:
        parts.append("Likely: " + ", ".join(likely) + ".")
    if possible:
        parts.append("Possible: " + ", ".join(possible) + ".")
    if not likely and not possible:
        parts.append("No condition looks likely from the findings entered.")
    parts.append(ACTION_LABEL[action] + ".")
    if referral and referral.get("ok") and action in ("refer", "refer_urgently"):
        parts.append(f"Go to {referral['destination_name']} (about {referral['minutes']:.0f} min).")
    return " ".join(parts)


def run_triage(model, findings, cost_settings=None, caps=None, graph=None, origin=None,
               kill_switch=False, use_mean=False, request_id="-"):
    """Run the whole pipeline. Never raises: any failure returns 'refer for clinician review'."""
    t0 = time.perf_counter()
    steps = {}
    cs = cost_settings or {}
    m_w, u_w = float(cs.get("missed_case_weight", 10)), float(cs.get("unneeded_referral_weight", 1))
    caps = set(caps if caps is not None else BASE_CAPS["SC"])

    t = time.perf_counter()
    v = validate(findings)
    steps["validate"] = (time.perf_counter() - t) * 1000
    if v["errors"]:
        return {"status": "invalid", "errors": v["errors"], "warnings": v["warnings"],
                "model_version": MODEL_VERSION, "disclaimer": DISCLAIMER, "degraded": False}
    clean = v["clean"]

    t = time.perf_counter()
    flags = red_flags(clean)
    steps["redflags"] = (time.perf_counter() - t) * 1000

    base = {"status": "ok", "model_version": MODEL_VERSION, "disclaimer": DISCLAIMER,
            "warnings": v["warnings"], "red_flags": flags, "degraded": False, "mode": "model"}

    if kill_switch:
        action, why = guideline_checklist(clean, flags)
        ref = plan_referral(graph, origin, action) if graph is not None and action != "treat_locally" else None
        col = RISK_UI[action]
        base.update({"mode": "guideline_checklist", "recommended_action": action, "risk_colour": col[0],
                     "risk_label": col[1], "risk_icon": col[2], "summary": why, "probabilities": {},
                     "tests": [], "referral": ref, "explanation": [], "confidence": "n/a",
                     "insufficient_evidence": False, "latency_ms": (time.perf_counter() - t0) * 1000})
        return base

    # Safety override: do not run Bayesian inference or routine test ranking when a red flag is present.
    # The referral pathway is surfaced immediately and cannot be displaced by model probabilities.
    if flags:
        action = "refer_urgently"
        col = RISK_UI[action]
        ref = plan_referral(graph, origin, action) if graph is not None else None
        details = "; ".join(flag["message"] for flag in flags)
        base.update({"mode": "red_flag_override", "recommended_action": action,
                     "risk_colour": col[0], "risk_label": col[1], "risk_icon": col[2],
                     "summary": f"Red flag present ({details}). Urgent referral takes priority; do not delay for model output or routine tests.",
                     "probabilities": {}, "state_probs": {}, "confidence": "n/a",
                     "insufficient_evidence": False, "interval_width": None,
                     "n_findings": len(encode(clean)), "action_eu": {}, "action_agreement": 0.0,
                     "sensitivity": None, "tests": [], "next_test": None, "referral": ref,
                     "explanation": [], "safety_notice": True,
                     "latency_ms": (time.perf_counter() - t0) * 1000, "step_ms": steps})
        _log(event="triage", request_id=request_id, model_version=MODEL_VERSION,
             action=action, degraded=False, red_flag_override=True,
             latency_ms=round(base["latency_ms"], 1))
        return base

    try:
        ev = encode(clean)
        if not ev:
            action = "refer_urgently" if flags else "one_more_test"
            col = RISK_UI[action]
            summary = ("A red-flag rule requires urgent referral; do not delay for model output." if flags else
                       "No model-usable findings were entered. Collect relevant findings before interpreting risk.")
            base.update({"mode": "model", "recommended_action": action,
                         "risk_colour": col[0], "risk_label": col[1], "risk_icon": col[2],
                         "summary": summary, "probabilities": {}, "state_probs": {},
                         "confidence": "n/a", "insufficient_evidence": True, "interval_width": None,
                         "n_findings": 0, "action_eu": {}, "action_agreement": 0.0,
                         "sensitivity": None, "tests": [], "next_test": None,
                         "referral": None, "explanation": [], "safety_notice": bool(flags),
                         "latency_ms": (time.perf_counter() - t0) * 1000})
            _log(event="triage", request_id=request_id, model_version=MODEL_VERSION,
                 action=action, degraded=False, no_model_evidence=True)
            return base
        t = time.perf_counter()
        post, _, _ = posterior(model, ev, use_mean)
        steps["bn"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        cp = cond_probs(post)
        probs = {c: summarize(cp[c]) for c in TARGETS}
        sp = state_probs(post)
        n_obs = len(ev)
        level, insufficient, width = _confidence([probs[c]["high80"] - probs[c]["low80"] for c in TARGETS], n_obs)
        steps["confidence"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        U = utility_matrix(m_w, u_w)
        P = sp.mean(axis=0)
        allow = list(ACTIONS)
        if insufficient:
            allow.remove("treat_locally")  # never "treat locally" on thin evidence
        action, eu = choose_action(P, U, allow)
        # how stable is the choice across the K table samples?
        picks = [choose_action(p, U, allow)[0] for p in sp[:: max(1, len(sp) // 50)]]
        agreement = picks.count(action) / len(picks)
        steps["utility"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        tests = rank_tests(model, ev, U, caps, use_mean)
        useful = [x for x in tests if x["net_voi"] > 0]
        if insufficient and tests and action not in ("refer", "refer_urgently"):
            action = "one_more_test"  # thin evidence: collect more findings first
            useful = useful or tests[:1]
        elif action == "one_more_test" and not useful:
            action, eu = choose_action(P, U, [a for a in allow if a != "one_more_test"])
        steps["voi"] = (time.perf_counter() - t) * 1000

        if flags:
            action = "refer_urgently"  # red flags override the model
            useful = []  # never advertise routine testing as the next action when urgent referral is required
        t = time.perf_counter()
        ref = None
        if action in ("refer", "refer_urgently") and graph is not None:
            ref = plan_referral(graph, origin, action, float(P[2]))
        steps["routing"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        expl = explain(model, ev, clean)
        steps["explain"] = (time.perf_counter() - t) * 1000

        col = RISK_UI[action]
        base.update({
            "probabilities": probs, "state_probs": dict(zip(RISK_STATES, [float(x) for x in P])),
            "confidence": level, "insufficient_evidence": insufficient, "interval_width": width,
            "n_findings": n_obs, "recommended_action": action,
            "action_eu": {a: float(e) for a, e in zip(ACTIONS, eu)}, "action_agreement": agreement,
            "sensitivity": action_sensitivity(P, u_w), "tests": tests,
            "next_test": (useful[0] if useful and action != "refer_urgently" else None),
            "referral": ref, "explanation": expl, "safety_notice": bool(flags),
            "risk_colour": col[0], "risk_label": col[1], "risk_icon": col[2],
            "summary": _summary(probs, action, ref, flags),
        })
    except Exception as exc:  # fail safe
        action = "refer_urgently" if flags else "refer"
        col = RISK_UI[action]
        base.update({"degraded": True, "mode": "degraded", "recommended_action": action,
                     "risk_colour": col[0], "risk_label": col[1], "risk_icon": col[2],
                     "summary": "Refer for clinician review (the engine could not complete).",
                     "probabilities": {}, "tests": [], "referral": None, "explanation": [],
                     "confidence": "n/a", "insufficient_evidence": True, "error_type": type(exc).__name__})
        log.error(json.dumps({"event": "degraded", "request_id": request_id, "error": type(exc).__name__}))
    base["latency_ms"] = (time.perf_counter() - t0) * 1000
    base["step_ms"] = steps
    _log(event="triage", request_id=request_id, model_version=MODEL_VERSION, action=base["recommended_action"],
         degraded=base["degraded"], latency_ms=round(base["latency_ms"], 1))
    return base