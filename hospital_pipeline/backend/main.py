"""
MEMBER 4 — FastAPI Backend
Job: Expose all pipeline data via REST API endpoints
Run: uvicorn main:app --reload
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import pandas as pd
import json
import subprocess
import sys

# ── This line is what uvicorn looks for ──────────────────────
app = FastAPI(title="Hospital Pipeline API", version="1.0.0")

# ── STATIC FILES — Serve frontend assets ────────────────────
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# ── CORS — allows frontend to call backend ───────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helper: read any CSV and return as list ───────────────────
def read_csv(path: str):
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{path} not found. Run the pipeline first.")
    df = pd.read_csv(p)
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")


@app.get("/")
def root():
    frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    try:
        with open(frontend_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return {"error": "Frontend not found. Please ensure frontend/index.html exists."}


@app.get("/api")
def api_root():
    return {"status": "Hospital Pipeline API is running!", "version": "1.0.0"}


@app.post("/pipeline/run")
def run_pipeline():
    try:
        script = Path(__file__).parent / "run_pipeline.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(__file__).parent)
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
        return {"success": True, "message": "Pipeline completed!", "output": result.stdout}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Pipeline took too long")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/status")
def pipeline_status():
    base = Path(__file__).parent
    files = {
        "bronze_ehr":     (base / "bronze/ehr.csv").exists(),
        "bronze_vitals":  (base / "bronze/vitals.csv").exists(),
        "bronze_labs":    (base / "bronze/labs.csv").exists(),
        "silver_vitals":  (base / "silver/clean_vitals.csv").exists(),
        "silver_master":  (base / "silver/patient_master.csv").exists(),
        "gold_anomalies": (base / "gold/anomalies.csv").exists(),
        "gold_summary":   (base / "gold/summary.json").exists(),
    }
    return {"files": files, "completed": sum(files.values()), "total": len(files), "ready": all(files.values())}


@app.get("/dashboard/summary")
def dashboard_summary():
    base = Path(__file__).parent
    summary = {"total_anomalies": 0, "critical": 0, "warning": 0,
                "patients_affected": 0, "total_patients": 0, "total_vitals": 0}
    summary_path = base / "gold/summary.json"
    if summary_path.exists():
        summary.update(json.loads(summary_path.read_text()))
    master_path = base / "silver/patient_master.csv"
    if master_path.exists():
        summary["total_patients"] = len(pd.read_csv(master_path))
    vitals_path = base / "silver/clean_vitals.csv"
    if vitals_path.exists():
        summary["total_vitals"] = len(pd.read_csv(vitals_path))
    return summary


@app.get("/patients")
def get_patients():
    base = Path(__file__).parent
    return read_csv(str(base / "silver/patient_master.csv"))


@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    records = get_patients()
    match = [r for r in records if r.get("patient_id") == patient_id]
    if not match:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return match[0]


@app.get("/anomalies")
def get_anomalies(severity: str = None, patient_id: str = None):
    base = Path(__file__).parent
    records = read_csv(str(base / "gold/anomalies.csv"))
    if severity:
        records = [r for r in records if r.get("severity") == severity]
    if patient_id:
        records = [r for r in records if r.get("patient_id") == patient_id]
    return records


@app.get("/stats/anomaly-types")
def anomaly_types():
    base = Path(__file__).parent
    p = base / "gold/anomalies.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    counts = df["anomaly_type"].value_counts().reset_index()
    counts.columns = ["type", "count"]
    return counts.to_dict(orient="records")


@app.get("/stats/vitals-trend")
def vitals_trend():
    base = Path(__file__).parent
    p = base / "silver/clean_vitals.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    df = df.where(pd.notnull(df), None)
    return df[["patient_id", "timestamp", "hr", "ox", "sys", "dia"]].to_dict(orient="records")
