from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import io
import os
import json
import datetime
from sklearn.linear_model import LogisticRegression

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, "data")
DEFAULT_DATASET = os.path.join(DATA_DIR, "sample_transactions.csv")
UPLOADED_DATASET = os.path.join(DATA_DIR, "uploaded_transactions.csv")
SMART_CFG_PATH = os.path.join(DATA_DIR, "smart_routing.json")

DATA_PATH = DEFAULT_DATASET

app = FastAPI(title="Smart Routing Lab (PoC v3)", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROVIDER_FEES = {
    "A": {"fixed": 0.10, "pct": 0.019},
    "B": {"fixed": 0.12, "pct": 0.017},
    "C": {"fixed": 0.09, "pct": 0.022},
}

class SimulationResult(BaseModel):
    approval_rate: float
    avg_fee: float
    p95_latency: float
    approved_volume: float
    total_attempted_volume: float
    total_transactions: int

class TrainSummary(BaseModel):
    status: str
    segments_count: int
    segments: List[Dict[str, Any]]

class RouteRequest(BaseModel):
    country: str
    card_brand: str
    amount: float
    device: str

def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

def fee_for(provider: str, amount: float) -> float:
    f = PROVIDER_FEES.get(provider, {"fixed": 0.0, "pct": 0.0})
    return float(f["fixed"] + f["pct"] * amount)

def compute_metrics(df: pd.DataFrame) -> SimulationResult:
    total_attempted = float(df["amount"].sum())
    approved_df = df[df["status"] == "approved"]
    approved_volume = float(approved_df["amount"].sum())
    approval_rate = float(len(approved_df) / len(df)) if len(df) else 0.0

    fees = [fee_for(str(row["provider"]), float(row["amount"])) for _, row in df.iterrows()]
    avg_fee = float(sum(fees) / len(fees)) if fees else 0.0

    p95_latency = float(df["latency_ms"].quantile(0.95)) if len(df) else 0.0

    return SimulationResult(
        approval_rate=approval_rate,
        avg_fee=avg_fee,
        p95_latency=p95_latency,
        approved_volume=approved_volume,
        total_attempted_volume=total_attempted,
        total_transactions=int(len(df)),
    )

def segment_id(country: str, brand: str) -> str:
    return f"{country}_{brand}"

def load_smart_cfg() -> Optional[Dict[str, Any]]:
    if not os.path.exists(SMART_CFG_PATH):
        return None
    with open(SMART_CFG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/baseline", response_model=SimulationResult)
def baseline_metrics():
    df = load_dataset()
    return compute_metrics(df)

@app.post("/upload", response_model=SimulationResult)
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(UPLOADED_DATASET, index=False)

    global DATA_PATH
    DATA_PATH = UPLOADED_DATASET

    if os.path.exists(SMART_CFG_PATH):
        os.remove(SMART_CFG_PATH)

    return compute_metrics(df)

@app.post("/train-smart", response_model=TrainSummary)
def train_smart_router() -> TrainSummary:
    df = load_dataset().copy()
    df["success"] = (df["status"] == "approved").astype(int)

    df["country_code"] = df["country"].astype("category").cat.codes
    df["brand_code"] = df["card_brand"].astype("category").cat.codes
    df["provider_code"] = df["provider"].astype("category").cat.codes
    df["device_code"] = df["device"].astype("category").cat.codes

    feature_cols = ["country_code", "brand_code", "device_code", "amount", "provider_code"]
    X = df[feature_cols]
    y = df["success"]

    model = LogisticRegression(max_iter=600)
    model.fit(X, y)

    providers = sorted(df["provider"].unique())

    segments_out: List[Dict[str, Any]] = []
    country_cats = df["country"].astype("category").cat.categories
    brand_cats = df["card_brand"].astype("category").cat.categories
    device_cats = df["device"].astype("category").cat.categories
    provider_cats = df["provider"].astype("category").cat.categories

    for (country, brand), seg_df in df.groupby(["country", "card_brand"]):
        ref = seg_df.iloc[0].copy()
        scored = []
        for p in providers:
            row_country_code = int(country_cats.get_loc(country))
            row_brand_code = int(brand_cats.get_loc(brand))
            row_device_code = int(device_cats.get_loc(str(ref["device"])))
            row_provider_code = int(provider_cats.get_loc(p))

            x_row = [[row_country_code, row_brand_code, row_device_code, float(ref["amount"]), row_provider_code]]
            p_success = float(model.predict_proba(x_row)[0][1])
            expected_fee = fee_for(p, float(ref["amount"]))
            score = p_success * float(ref["amount"]) - expected_fee

            scored.append({
                "provider": p,
                "p_success": p_success,
                "expected_fee": expected_fee,
                "score": score,
            })

        scored_sorted = sorted(scored, key=lambda r: r["score"], reverse=True)
        segments_out.append({
            "segment": segment_id(str(country), str(brand)),
            "country": str(country),
            "card_brand": str(brand),
            "routing_order": [r["provider"] for r in scored_sorted],
            "details": scored_sorted,
        })

    cfg = {"generated_at": datetime.datetime.utcnow().isoformat() + "Z", "segments": segments_out}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SMART_CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    return TrainSummary(status="ok", segments_count=len(segments_out), segments=segments_out)

@app.post("/route")
def route_payment(req: RouteRequest) -> Dict[str, Any]:
    cfg = load_smart_cfg()
    if not cfg:
        return {"error": "Smart router not trained yet. Call /train-smart first."}

    seg = segment_id(req.country, req.card_brand)
    segment = next((s for s in cfg["segments"] if s["segment"] == seg), None)
    if not segment:
        return {"error": f"No smart routing segment for {seg}"}

    primary = segment["routing_order"][0]
    reason = f"Segment {seg}: primary provider '{primary}' maximizes (predicted approval × amount − fees) on your dataset."
    return {
        "provider_order": segment["routing_order"],
        "primary_provider": primary,
        "segment": seg,
        "reason": reason,
    }

def learned_segment_approval_map(df: pd.DataFrame) -> Dict[Tuple[str, str, str], float]:
    out: Dict[Tuple[str, str, str], float] = {}
    for (country, brand, provider), g in df.groupby(["country", "card_brand", "provider"]):
        out[(str(country), str(brand), str(provider))] = float((g["status"] == "approved").mean())
    return out

@app.get("/simulate-smart", response_model=SimulationResult)
def simulate_smart():
    cfg = load_smart_cfg()
    df = load_dataset().copy()

    if not cfg:
        return compute_metrics(df)

    seg_to_primary = {s["segment"]: s["routing_order"][0] for s in cfg["segments"]}
    rates = learned_segment_approval_map(df)

    import random
    random.seed(123)

    new_rows = []
    for _, row in df.iterrows():
        seg = segment_id(str(row["country"]), str(row["card_brand"]))
        chosen = seg_to_primary.get(seg, str(row["provider"]))
        p = rates.get((str(row["country"]), str(row["card_brand"]), chosen), 0.5)

        success = random.random() < p
        status = "approved" if success else "declined"

        latency = float(row["latency_ms"])
        if chosen == "B":
            latency *= 0.92
        elif chosen == "C":
            latency *= 1.05

        d = row.to_dict()
        d["provider"] = chosen
        d["status"] = status
        d["latency_ms"] = int(latency)
        new_rows.append(d)

    new_df = pd.DataFrame(new_rows)
    return compute_metrics(new_df)

@app.get("/smart-config")
def smart_config() -> Dict[str, Any]:
    cfg = load_smart_cfg()
    if not cfg:
        return {"error": "Smart router not trained yet. Call /train-smart first."}
    return cfg

def compute_report_payload() -> Dict[str, Any]:
    baseline = baseline_metrics().model_dump()
    smart = simulate_smart().model_dump()
    cfg = load_smart_cfg() or {"segments": []}

    d_approval_pts = (smart["approval_rate"] - baseline["approval_rate"]) * 100.0
    d_approved_vol = smart["approved_volume"] - baseline["approved_volume"]
    d_fee = smart["avg_fee"] - baseline["avg_fee"]
    d_p95 = smart["p95_latency"] - baseline["p95_latency"]

    segments = cfg.get("segments", [])
    top_segments = segments[:20]

    return {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "dataset": os.path.basename(DATA_PATH),
        "baseline": baseline,
        "smart": smart,
        "deltas": {
            "approval_pts": d_approval_pts,
            "approved_volume": d_approved_vol,
            "avg_fee": d_fee,
            "p95_latency": d_p95,
        },
        "top_segments": top_segments,
    }

def render_report_html(payload: Dict[str, Any]) -> str:
    b = payload["baseline"]
    s = payload["smart"]
    d = payload["deltas"]

    def pct(x): return f"{x*100:.1f}%"
    def eur(x): return f"{x:,.0f} €"
    def eur3(x): return f"{x:.3f} €"
    def ms(x): return f"{x:.0f} ms"
    def sgn(x): return ("+" if x >= 0 else "") + f"{x:.1f}"

    rows = ""
    for seg in payload["top_segments"]:
        best = seg.get("routing_order", ["-"])[0]
        pbest = seg.get("details", [{}])[0].get("p_success", None)
        fee = seg.get("details", [{}])[0].get("expected_fee", None)
        order = " → ".join(seg.get("routing_order", []))
        rows += f"""
          <tr>
            <td>{seg.get("segment","-")}</td>
            <td><b>{best}</b></td>
            <td>{(pbest*100):.1f}%</td>
            <td>{fee:.3f} €</td>
            <td>{order}</td>
          </tr>
        """ if pbest is not None and fee is not None else f"""
          <tr>
            <td>{seg.get("segment","-")}</td>
            <td><b>{best}</b></td>
            <td>-</td><td>-</td>
            <td>{order}</td>
          </tr>
        """

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Smart Routing Lab Report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#fbfbfb; margin:0; }}
    .wrap {{ max-width: 980px; margin: 0 auto; padding: 28px; }}
    .hero {{ background:#fff; border:1px solid #eee; border-radius:16px; padding:18px 18px 14px; }}
    .badges span {{ display:inline-block; border:1px solid #e5e5e5; border-radius:999px; padding:2px 10px; font-size:12px; opacity:.85; margin-right:8px; }}
    h1 {{ margin:0; font-size:26px; }}
    .sub {{ margin:8px 0 0; font-size:13px; opacity:.8; }}
    .grid {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:12px; margin-top:14px; }}
    .kpi {{ background:#fff; border:1px solid #eee; border-radius:14px; padding:14px; }}
    .kpi .lbl {{ font-size:12px; opacity:.7; text-transform:uppercase; }}
    .kpi .val {{ font-size:20px; font-weight:650; margin-top:4px; }}
    .kpi .delta {{ display:inline-block; margin-top:8px; border:1px solid #e5e5e5; border-radius:999px; padding:2px 8px; font-size:12px; }}
    .card {{ background:#fff; border:1px solid #eee; border-radius:16px; padding:16px; margin-top:14px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ text-align:left; padding:10px; border-bottom:1px solid #f0f0f0; white-space:nowrap; }}
    th {{ background:#fafafa; border-bottom:1px solid #eee; }}
    .foot {{ margin-top:14px; font-size:12px; opacity:.65; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="badges">
        <span>Smart Routing Lab</span>
        <span>Report</span>
        <span>Dataset: {payload["dataset"]}</span>
        <span>Generated: {payload["generated_at"]}</span>
      </div>
      <h1>Baseline vs Smart Uplift</h1>
      <p class="sub">This report compares baseline KPIs from the dataset against a simulated smart-routing policy learned from historical outcomes.</p>

      <div class="grid">
        <div class="kpi">
          <div class="lbl">Approval rate</div>
          <div class="val">{pct(s["approval_rate"])}</div>
          <div class="delta">Δ {sgn(d["approval_pts"])} pts</div>
        </div>
        <div class="kpi">
          <div class="lbl">Approved volume</div>
          <div class="val">{eur(s["approved_volume"])}</div>
          <div class="delta">Δ {("+" if d["approved_volume"]>=0 else "")}{eur(d["approved_volume"])}</div>
        </div>
        <div class="kpi">
          <div class="lbl">Avg fee per tx</div>
          <div class="val">{eur3(s["avg_fee"])}</div>
          <div class="delta">Δ {("+" if d["avg_fee"]>=0 else "")}{eur3(d["avg_fee"])}</div>
        </div>
        <div class="kpi">
          <div class="lbl">p95 latency</div>
          <div class="val">{ms(s["p95_latency"])}</div>
          <div class="delta">Δ {("+" if d["p95_latency"]>=0 else "")}{ms(d["p95_latency"])}</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h2 style="margin:0 0 10px; font-size:16px;">Key numbers</h2>
      <table>
        <tr><th></th><th>Baseline</th><th>Smart (simulated)</th></tr>
        <tr><td>Approval rate</td><td>{pct(b["approval_rate"])}</td><td>{pct(s["approval_rate"])}</td></tr>
        <tr><td>Approved volume</td><td>{eur(b["approved_volume"])}</td><td>{eur(s["approved_volume"])}</td></tr>
        <tr><td>Avg fee / tx</td><td>{eur3(b["avg_fee"])}</td><td>{eur3(s["avg_fee"])}</td></tr>
        <tr><td>p95 latency</td><td>{ms(b["p95_latency"])}</td><td>{ms(s["p95_latency"])}</td></tr>
        <tr><td>Total tx</td><td>{b["total_transactions"]}</td><td>{s["total_transactions"]}</td></tr>
      </table>
    </div>

    <div class="card">
      <h2 style="margin:0 0 10px; font-size:16px;">Top segments (learned policy)</h2>
      <table>
        <thead>
          <tr>
            <th>Segment</th>
            <th>Primary</th>
            <th>Pred. approval</th>
            <th>Est. fee</th>
            <th>Provider order</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>

    <div class="foot">PoC v3. For demo purposes only. Exported by /report?format=html|pdf.</div>
  </div>
</body>
</html>"""
    return html

def render_report_pdf(payload: Dict[str, Any]) -> bytes:
    b = payload["baseline"]
    s = payload["smart"]
    d = payload["deltas"]

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    def txt(x_mm, y_mm, text, size=10, bold=False):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x_mm * mm, y_mm * mm, text)

    # Header
    txt(18, 285, "Smart Routing Lab", 16, True)
    txt(18, 277, "Baseline vs Smart Uplift Report", 12, True)
    txt(18, 270, f"Dataset: {payload['dataset']}   Generated: {payload['generated_at']}", 9)

    # KPI block
    y = 258
    txt(18, y, f"Smart approval rate: {s['approval_rate']*100:.1f}%   (Δ {d['approval_pts']:+.1f} pts)", 11, True)
    y -= 8
    txt(18, y, f"Smart approved volume: {s['approved_volume']:.0f} €   (Δ {d['approved_volume']:+.0f} €)", 10)
    y -= 6
    txt(18, y, f"Smart avg fee/tx: {s['avg_fee']:.3f} €   (Δ {d['avg_fee']:+.3f} €)", 10)
    y -= 6
    txt(18, y, f"Smart p95 latency: {s['p95_latency']:.0f} ms   (Δ {d['p95_latency']:+.0f} ms)", 10)

    # Baseline vs Smart table
    y -= 14
    txt(18, y, "Key numbers", 11, True)
    y -= 8
    txt(18, y, "Metric", 9, True); txt(70, y, "Baseline", 9, True); txt(110, y, "Smart", 9, True)
    y -= 6
    c.line(18*mm, y*mm, 190*mm, y*mm)
    y -= 8

    rows = [
        ("Approval rate", f"{b['approval_rate']*100:.1f}%", f"{s['approval_rate']*100:.1f}%"),
        ("Approved volume", f"{b['approved_volume']:.0f} €", f"{s['approved_volume']:.0f} €"),
        ("Avg fee / tx", f"{b['avg_fee']:.3f} €", f"{s['avg_fee']:.3f} €"),
        ("p95 latency", f"{b['p95_latency']:.0f} ms", f"{s['p95_latency']:.0f} ms"),
        ("Total tx", str(b["total_transactions"]), str(s["total_transactions"])),
    ]
    for r in rows:
        txt(18, y, r[0], 9); txt(70, y, r[1], 9); txt(110, y, r[2], 9)
        y -= 7
        if y < 60:
            c.showPage()
            y = 285

    # Segments
    y -= 6
    txt(18, y, "Top segments (learned policy)", 11, True)
    y -= 8
    txt(18, y, "Segment", 9, True); txt(70, y, "Primary", 9, True); txt(95, y, "Pred", 9, True); txt(120, y, "Fee", 9, True)
    y -= 6
    c.line(18*mm, y*mm, 190*mm, y*mm)
    y -= 8

    for seg in payload["top_segments"][:18]:
        best = (seg.get("routing_order") or ["-"])[0]
        pbest = None
        fee = None
        if seg.get("details") and seg["details"][0].get("p_success") is not None:
            pbest = seg["details"][0]["p_success"] * 100.0
        if seg.get("details") and seg["details"][0].get("expected_fee") is not None:
            fee = seg["details"][0]["expected_fee"]
        txt(18, y, seg.get("segment","-")[:18], 9)
        txt(70, y, str(best), 9)
        txt(95, y, "-" if pbest is None else f"{pbest:.1f}%", 9)
        txt(120, y, "-" if fee is None else f"{fee:.3f}€", 9)
        y -= 7
        if y < 35:
            break

    txt(18, 18, "PoC v3. Exported by /report?format=pdf|html", 8)
    c.showPage()
    c.save()
    return buf.getvalue()

@app.get("/report")
def report(format: str = "pdf"):
    """Generate a one-click shareable report.

    - format=pdf returns an attachment PDF
    - format=html returns an HTML page
    """
    payload = compute_report_payload()

    if format.lower() == "html":
        html = render_report_html(payload)
        return HTMLResponse(content=html)

    if format.lower() == "json":
        return JSONResponse(content=payload)

    pdf_bytes = render_report_pdf(payload)
    filename = f"smart-routing-report-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.pdf"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf", headers=headers)
