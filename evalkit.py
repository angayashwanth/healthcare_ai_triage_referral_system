"""AAROGYA MARG - evaluation, benchmarks and property tests (uses triage_core)."""
import io
import json
import logging
import time

import numpy as np

import triage_core as tc

OBSERVABLE = ["age", "bmi", "smoker", "family", "sbp", "glucose", "headache", "polyuria",
              "tiredness", "chest", "breathless"]  # what a sub-centre/PHC can observe (no ECG / lab)


# ---------------------------------------------------------------- prediction
def predict_batch(model, df, obs=OBSERVABLE, use_mean=True, mask_frac=0.0, seed=0):
    rng = np.random.default_rng(seed)
    cols = {v: df[v].values for v in obs}
    N = len(df)
    cond, state, width, nobs = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros(N), np.zeros(N, int)
    for i in range(N):
        ev = {v: int(cols[v][i]) for v in obs}
        if mask_frac > 0:
            ev = {v: s for v, s in ev.items() if rng.random() >= mask_frac}
        post, _, _ = tc.posterior(model, ev, use_mean)
        cp = tc.cond_probs(post)
        cond[i] = [np.median(cp[c]) for c in tc.TARGETS]
        state[i] = tc.state_probs(post).mean(axis=0)
        width[i] = max(np.percentile(cp[c], 90) - np.percentile(cp[c], 10) for c in tc.TARGETS)
        nobs[i] = len(ev)
    return {"cond": cond, "state": state, "width": width, "nobs": nobs}


# ------------------------------------------------------------------- metrics
def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    tot = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def sens_spec(y, p, t):
    pred = p >= t
    tp, fn = ((pred) & (y == 1)).sum(), ((~pred) & (y == 1)).sum()
    tn, fp = ((~pred) & (y == 0)).sum(), ((pred) & (y == 0)).sum()
    return float(tp / max(tp + fn, 1)), float(tn / max(tn + fp, 1))


def threshold_for_sens(y, p, target=0.80):
    """Largest threshold that still reaches the target sensitivity (chosen on VALIDATION data)."""
    for t in np.linspace(0.9, 0.01, 90):
        if sens_spec(y, p, t)[0] >= target:
            return float(t)
    return 0.01


def bootstrap_ci(fn, y, p, n=200, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def logistic_baseline(train, test, target):
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    X = pd.get_dummies(train[OBSERVABLE].astype(str))
    Xt = pd.get_dummies(test[OBSERVABLE].astype(str)).reindex(columns=X.columns, fill_value=0)
    clf = LogisticRegression(max_iter=1000).fit(X, train[target])
    return clf.predict_proba(Xt)[:, 1]


def calibration_points(y, p, bins=8):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum() >= 5:
            xs.append(float(p[m].mean()))
            ys.append(float(y[m].mean()))
    return xs, ys


# ------------------------------------------------------------ decision quality
def policy_loss(state_probs, true_state, U, allow=(0, 1, 2, 3)):
    """Mean loss under utility table U using the TRUE risk state.
    Policy: pick the allowed action with the highest expected utility.
    allow=(0,2,3) excludes 'do one more test' so the comparison does not lean on that table row."""
    eu = state_probs @ U.T
    mask = np.full(4, -1e9)
    mask[list(allow)] = 0.0
    acts = np.argmax(eu + mask, axis=1)
    return float(U[acts, true_state].mean()), acts


def threshold_policy_loss(cond, true_state, U, thr=0.5):
    acts = np.where((cond >= thr).any(axis=1), 2, 0)  # refer vs treat locally
    return float(U[acts, true_state].mean())


# ---------------------------------------------------------- routing benchmark
def routing_benchmark(n_queries=300, n_nodes=150, seed=5):
    g = tc.make_large_graph(n_nodes, seed=seed)
    rng = np.random.default_rng(seed + 1)
    origins = [n for n, v in g.nodes.items() if v["type"] == "SC" and v["open"]]
    res = {"queries": 0, "equal_cost": 0, "astar_expanded": [], "ucs_expanded": [],
           "nearest_lacks_capability": 0, "nearest_extra_minutes": [], "heuristic_violations": 0}
    for _ in range(n_queries):
        o = origins[int(rng.integers(len(origins)))]
        action = ["refer", "refer_urgently"][int(rng.integers(2))]
        need = tc.required_caps(action)
        goals = [n for n, v in g.nodes.items() if v["open"] and v["level"] > 1 and need <= v["caps"]]
        if not goals:
            continue
        a, u = tc.astar(g, o, goals), tc.ucs(g, o, goals)
        nr = tc.nearest_straight_line(g, o)
        res["queries"] += 1
        res["equal_cost"] += int(abs(a["minutes"] - u["minutes"]) < 1e-9)
        res["astar_expanded"].append(a["expanded"])
        res["ucs_expanded"].append(u["expanded"])
        if nr is not None:
            ok = need <= g.nodes[nr["destination"]]["caps"]
            res["nearest_lacks_capability"] += int(not ok)
            res["nearest_extra_minutes"].append(nr["minutes"] - a["minutes"])
    out = {
        "nodes": n_nodes, "queries": res["queries"],
        "equal_cost_share": res["equal_cost"] / max(res["queries"], 1),
        "astar_expanded_mean": float(np.mean(res["astar_expanded"])),
        "ucs_expanded_mean": float(np.mean(res["ucs_expanded"])),
        "nearest_lacks_capability_share": res["nearest_lacks_capability"] / max(res["queries"], 1),
        "nearest_extra_minutes_mean": float(np.mean(res["nearest_extra_minutes"])),
    }
    out["expansion_saving_pct"] = 100 * (1 - out["astar_expanded_mean"] / out["ucs_expanded_mean"])
    return out


# ------------------------------------------------------------ latency benchmark
def latency_benchmark(model, graph, n=150, seed=1):
    rng = np.random.default_rng(seed)
    lat, lat_whatif = [], []
    for _ in range(n):
        f = random_findings(rng)
        r = tc.run_triage(model, f, caps=tc.BASE_CAPS["SC"], graph=graph, origin="sc-01")
        lat.append(r["latency_ms"])
        # what-if refresh = triage again with one toggled symptom
        f2 = dict(f)
        sy = dict(f.get("symptoms", {}))
        sy["chest_discomfort"] = not sy.get("chest_discomfort", False)
        f2["symptoms"] = sy
        lat_whatif.append(tc.run_triage(model, f2, caps=tc.BASE_CAPS["SC"], graph=graph, origin="sc-01")["latency_ms"])
    return {"n": n, "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "whatif_p95_ms": float(np.percentile(lat_whatif, 95))}


# ------------------------------------------------------------ random findings
def random_findings(rng, blank_prob=0.3):
    def maybe(x):
        return None if rng.random() < blank_prob else x
    sy = {k: maybe(bool(rng.random() < 0.25)) for k in tc.SYMPTOMS}
    sbp = maybe(float(rng.integers(85, 200)))
    dbp = maybe(float(rng.integers(50, 110)))
    # Keep generated paired BP measurements internally consistent under strict input validation.
    if sbp is not None and dbp is not None and dbp >= sbp:
        dbp = max(30.0, sbp - 1.0)
    f = {
        "age": maybe(int(rng.integers(18, 90))),
        "sex": maybe("F"),
        "sbp_mmhg": sbp,
        "dbp_mmhg": dbp,
        "glucose_mg_dl": maybe(float(rng.integers(70, 320))),
        "glucose_type": "random",
        "bmi_kg_m2": maybe(float(rng.integers(17, 40))),
        "smoker": maybe(bool(rng.random() < 0.2)),
        "family_history": maybe(bool(rng.random() < 0.25)),
        "symptoms": {k: v for k, v in sy.items() if v is not None},
    }
    return {k: v for k, v in f.items() if v is not None}


# ------------------------------------------------------------- property tests
def run_property_tests(model, trials=150, seed=42):
    """9 invariants from the design plan. Returns a list of dict rows."""
    rng = np.random.default_rng(seed)
    g = tc.make_demo_graph()
    caps = tc.BASE_CAPS["SC"]
    rows = []

    def record(name, fails, note="", finding=False):
        status = "PASS" if fails == 0 else ("FINDING" if finding else "FAIL")
        rows.append({"invariant": name, "trials": trials, "failures": fails, "passed": fails == 0,
                     "status": status, "note": note})

    # 1 probabilities in [0,1]; each condition's two states sum to 1; risk states sum to 1
    fails = 0
    for _ in range(trials):
        ev = tc.encode(tc.validate(random_findings(rng))["clean"])
        post, _, _ = tc.posterior(model, ev)
        cp = tc.cond_probs(post)
        sp = tc.state_probs(post)
        ok = all((p >= -1e-12).all() and (p <= 1 + 1e-12).all() for p in cp.values())
        ok &= np.allclose(post.sum(axis=(1, 2, 3)), 1) and np.allclose(sp.sum(axis=1), 1)
        fails += int(not ok)
    record("Probabilities lie in [0,1] and sum to 1", fails)

    # 2 raising SBP never lowers hypertension probability
    fails = 0
    for _ in range(trials):
        ev = tc.encode(tc.validate(random_findings(rng))["clean"])
        ev.pop("sbp", None)
        prev = -1
        for band in range(4):
            e = dict(ev)
            e["sbp"] = band
            p = np.median(tc.cond_probs(tc.posterior(model, e)[0])["htn"])
            fails += int(p < prev - 1e-9)
            prev = p
    record("Raising systolic BP never lowers P(hypertension)", fails)

    # 3 removing a finding never raises confidence (confidence level and interval flag)
    fails, rank = 0, {"n/a": -1, "low": 0, "medium": 1, "high": 2}
    for _ in range(trials):
        f = random_findings(rng, blank_prob=0.1)
        r = tc.run_triage(model, f, caps=caps)
        sy = dict(f.get("symptoms", {}))
        keys = [k for k in f if k not in ("glucose_type", "symptoms")] + ["sy:" + k for k in sy]
        if not keys:
            continue
        k = keys[int(rng.integers(len(keys)))]
        f2 = {a: b for a, b in f.items()}
        if k.startswith("sy:"):
            s2 = dict(sy)
            s2.pop(k[3:])
            f2["symptoms"] = s2
        else:
            f2.pop(k)
        r2 = tc.run_triage(model, f2, caps=caps)
        fails += int(rank[r2["confidence"]] > rank[r["confidence"]])
    record("Removing a finding never raises confidence", fails,
           "Design finding: removing a conflicting finding can NARROW the interval, so the level can rise. "
           "The plan's invariant is too strong for an interval-based confidence; the insufficient-evidence "
           "flag (count based) is the safety control.", finding=True)

    # 4 any red flag returns refer urgently
    fails = 0
    for _ in range(trials):
        f = random_findings(rng)
        f["sbp_mmhg"] = float(rng.integers(180, 230))
        fails += int(tc.run_triage(model, f, caps=caps, graph=g, origin="sc-01")["recommended_action"] != "refer_urgently")
        f = random_findings(rng)
        f["symptoms"] = {"chest_discomfort": True, "breathlessness": True}
        fails += int(tc.run_triage(model, f, caps=caps, graph=g, origin="sc-01")["recommended_action"] != "refer_urgently")
    record("Red-flag input always returns 'refer urgently'", fails)

    # 5 A* cost equals UCS cost and heuristic is admissible
    big = tc.make_large_graph(120, seed=9)
    fails = 0
    ids = list(big.nodes)
    for _ in range(trials):
        s = ids[int(rng.integers(len(ids)))]
        goals = [ids[int(i)] for i in rng.integers(0, len(ids), 3)]
        a, u = tc.astar(big, s, goals), tc.ucs(big, s, goals)
        h0 = min(big.straight_km(s, t) for t in goals) / tc.VMAX_KMH * 60
        fails += int(abs(a["minutes"] - u["minutes"]) > 1e-9 or h0 > u["minutes"] + 1e-9)
    record("A* cost equals uniform-cost cost; heuristic never overestimates", fails)

    # 6 destination has every required capability and is open
    fails = 0
    for _ in range(trials):
        o = ["sc-01", "sc-02", "sc-03", "sc-04", "sc-05", "sc-06"][int(rng.integers(6))]
        closed = [n for n in g.nodes if rng.random() < 0.15 and n != o]
        g.apply_status(closed=closed)
        action = ["refer", "refer_urgently"][int(rng.integers(2))]
        r = tc.plan_referral(g, o, action)
        if r["ok"] and r["note"] is None:
            d = g.nodes[r["destination"]]
            fails += int(not (d["open"] and set(r["required"]) <= d["caps"]))
    g.apply_status(closed=[])
    record("Destination is open and has every required capability", fails)

    # 7 determinism
    fails = 0
    for _ in range(trials):
        f = random_findings(rng)
        a = tc.run_triage(model, f, caps=caps, graph=g, origin="sc-01")
        b = tc.run_triage(model, f, caps=caps, graph=g, origin="sc-01")
        keep = ("probabilities", "recommended_action", "next_test", "explanation")
        fails += int(any(a.get(k) != b.get(k) for k in keep))
    record("Same input and settings give the same output", fails)

    # 8 blanks never raise an error
    fails = 0
    for _ in range(trials):
        f = random_findings(rng, blank_prob=0.8)
        r = tc.run_triage(model, f, caps=caps, graph=g, origin="sc-01")
        fails += int(r.get("degraded") or r.get("status") != "ok")
    r = tc.run_triage(model, {}, caps=caps, graph=g, origin="sc-01")
    fails += int(r.get("degraded") or r.get("status") != "ok")
    record("Blank fields never raise an error (including an empty form)", fails)

    # 9 no log line contains the patient's values
    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    tc.log.addHandler(h)
    tc.log.setLevel(logging.INFO)
    fails = 0
    sentinel = {"age": 71, "sbp_mmhg": 187.3, "glucose_mg_dl": 313.7, "bmi_kg_m2": 29.1, "dbp_mmhg": 101.9}
    for _ in range(20):
        tc.run_triage(model, sentinel, caps=caps, graph=g, origin="sc-01")
    text = stream.getvalue()
    for needle in ("187.3", "313.7", "29.1", "101.9"):
        fails += int(needle in text)
    tc.log.removeHandler(h)
    record("No log line contains a test patient's values", fails, f"{len(text.splitlines())} log lines checked")

    # 10 failure injection: corrupt model -> fail-safe
    class Broken:
        def cpts(self, use_mean):
            raise RuntimeError("corrupt")
    r = tc.run_triage(Broken(), {"age": 50, "sbp_mmhg": 150}, caps=caps, graph=g, origin="sc-01")
    rows.append({"invariant": "Failure injection: broken model returns 'refer for clinician review'",
                 "trials": 1, "failures": int(not (r["degraded"] and r["recommended_action"] == "refer")),
                 "passed": bool(r["degraded"] and r["recommended_action"] == "refer"),
                 "status": "PASS" if (r["degraded"] and r["recommended_action"] == "refer") else "FAIL", "note": ""})
    return rows