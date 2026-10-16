"""AAROGYA MARG - local SQLite store (Tier 1). Append-only assessments, hash-chained audit log.
No names, phone numbers or government IDs are stored."""
import hashlib
import json
import sqlite3
import threading
import time
import uuid

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments(
  id TEXT PRIMARY KEY, patient_id TEXT, facility_id TEXT, created_by TEXT,
  findings TEXT, created_at REAL, model_version TEXT);
CREATE TABLE IF NOT EXISTS assessment_results(
  assessment_id TEXT PRIMARY KEY, action TEXT, risk_colour TEXT, probabilities TEXT,
  red_flags TEXT, referral TEXT, degraded INTEGER, mode TEXT, latency_ms REAL);
CREATE TABLE IF NOT EXISTS decisions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, assessment_id TEXT, decided_by TEXT, accepted INTEGER,
  override_reason TEXT, needs_mo_review INTEGER, decided_at REAL);
CREATE TABLE IF NOT EXISTS facility_status_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT, facility_id TEXT, status TEXT, detail TEXT, set_by TEXT, set_at REAL);
CREATE TABLE IF NOT EXISTS audit_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, actor TEXT, action TEXT, entity TEXT, entity_id TEXT,
  prev_hash TEXT, hash TEXT);
"""


class Store:
    def __init__(self, path="aarogya.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)
            self.db.commit()

    # -------------------------------------------------------------- audit
    def _audit(self, actor, action, entity, entity_id):
        row = self.db.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        prev = row[0] if row else "0" * 64
        ts = time.time()
        h = hashlib.sha256(f"{prev}|{ts}|{actor}|{action}|{entity}|{entity_id}".encode()).hexdigest()
        self.db.execute("INSERT INTO audit_log(ts,actor,action,entity,entity_id,prev_hash,hash) VALUES(?,?,?,?,?,?,?)",
                        (ts, actor, action, entity, entity_id, prev, h))

    def verify_chain(self):
        # A hash chain detects inconsistency; it is not immutable storage against DB rewrites.
        with self.lock:
            rows = self.db.execute("SELECT ts,actor,action,entity,entity_id,prev_hash,hash FROM audit_log ORDER BY id").fetchall()
        prev = "0" * 64
        for ts, actor, action, entity, eid, ph, h in rows:
            exp = hashlib.sha256(f"{prev}|{ts}|{actor}|{action}|{entity}|{eid}".encode()).hexdigest()
            if ph != prev or h != exp:
                return False, len(rows)
            prev = h
        return True, len(rows)

    # -------------------------------------------------------- assessments
    def save_assessment(self, facility_id, findings, result, created_by="demo-worker"):
        aid = "a_" + uuid.uuid4().hex[:10]
        pid = "p_" + uuid.uuid4().hex[:8]  # random, pseudonymous
        with self.lock:
            try:
                self.db.execute("INSERT INTO assessments VALUES(?,?,?,?,?,?,?)",
                                (aid, pid, facility_id, created_by, json.dumps(findings, default=str), time.time(),
                                 result.get("model_version")))
                self.db.execute("INSERT INTO assessment_results VALUES(?,?,?,?,?,?,?,?,?)",
                                (aid, result.get("recommended_action"), result.get("risk_colour"),
                                 json.dumps(result.get("probabilities")), json.dumps(result.get("red_flags")),
                                 json.dumps({k: v for k, v in (result.get("referral") or {}).items()
                                             if k in ("destination", "minutes", "km", "algorithm")}),
                                 int(result.get("degraded", False)), result.get("mode"), result.get("latency_ms")))
                self._audit(created_by, "create", "assessment", aid)
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
        return aid

    def add_decision(self, assessment_id, accepted, reason, red_flag_present, action, decided_by="demo-worker"):
        """Save one final decision per assessment; duplicate clicks are rejected atomically."""
        if not accepted and not reason:
            raise ValueError("A reason is required when overriding.")
        needs_review = int((not accepted) and (red_flag_present or action == "refer_urgently"))
        with self.lock:
            try:
                self.db.execute("BEGIN IMMEDIATE")
                exists = self.db.execute("SELECT 1 FROM assessments WHERE id=?", (assessment_id,)).fetchone()
                if not exists:
                    raise ValueError("Assessment not found; run triage again before saving a decision.")
                prior = self.db.execute("SELECT 1 FROM decisions WHERE assessment_id=? LIMIT 1", (assessment_id,)).fetchone()
                if prior:
                    raise ValueError("A decision has already been saved for this assessment. Run a new assessment to create a new decision.")
                self.db.execute("INSERT INTO decisions(assessment_id,decided_by,accepted,override_reason,needs_mo_review,decided_at)"
                                " VALUES(?,?,?,?,?,?)",
                                (assessment_id, decided_by, int(accepted), reason, needs_review, time.time()))
                self._audit(decided_by, "accept" if accepted else ("override_NEEDS_MO_REVIEW" if needs_review else "override"),
                            "assessment", assessment_id)
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
        return bool(needs_review)

    def has_decision(self, assessment_id):
        with self.lock:
            return self.db.execute("SELECT 1 FROM decisions WHERE assessment_id=? LIMIT 1", (assessment_id,)).fetchone() is not None

    # ----------------------------------------------------------- facilities
    def set_facility_status(self, closed, no_caps, set_by="demo-admin"):
        """Replace the current status with this snapshot (history is kept as log rows)."""
        with self.lock:
            try:
                self.db.execute("INSERT INTO facility_status_log(facility_id,status,detail,set_by,set_at) VALUES(?,?,?,?,?)",
                                ("*", "snapshot", json.dumps({"closed": sorted(closed),
                                                             "no_caps": {k: sorted(v) for k, v in no_caps.items()}}),
                                 set_by, time.time()))
                self._audit(set_by, "status_change", "facility", "*")
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise

    def current_status(self):
        with self.lock:
            row = self.db.execute("SELECT detail,set_at FROM facility_status_log ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            return [], {}, None
        d = json.loads(row[0])
        return d["closed"], {k: set(v) for k, v in d["no_caps"].items()}, row[1]

    # --------------------------------------------------------------- views
    def assessments_df(self):
        return pd.read_sql_query(
            "SELECT a.id, a.facility_id, datetime(a.created_at,'unixepoch') AS created_utc, r.action, r.risk_colour,"
            " r.mode, r.degraded, ROUND(r.latency_ms,1) AS latency_ms, d.accepted, d.override_reason, d.needs_mo_review"
            " FROM assessments a JOIN assessment_results r ON r.assessment_id=a.id"
            " LEFT JOIN decisions d ON d.assessment_id=a.id ORDER BY a.created_at DESC", self.db)

    def audit_df(self):
        return pd.read_sql_query("SELECT id, datetime(ts,'unixepoch') AS time_utc, actor, action, entity, entity_id,"
                                 " substr(hash,1,12) AS hash FROM audit_log ORDER BY id DESC", self.db)