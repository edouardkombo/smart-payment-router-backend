
# Smart Payment Router Backend

A FastAPI proof of concept that trains a simple "smart payment routing" policy from historical transaction outcomes, compares baseline KPIs vs a simulated smart-routing scenario, exposes a routing API, and exports a shareable report (HTML, PDF, or JSON).

This repository is meant for demos, internal labs, and product conversations. It is not a production-ready payment router.

## Why this exists (real payment-company context)

Payment companies and merchants commonly operate with multiple PSPs or acquirers. In real life, you see problems like:

- Approval rates vary by country, card brand, device, and provider.
- Providers have different fee schedules (fixed + percentage).
- Latency differences affect checkout conversion and timeouts.
- A provider can degrade or partially outage without warning.
- A "one provider for everything" setup leaves money on the table when performance diverges by segment.

A smart router is an attempt to answer, per payment attempt:

> Which provider should we try first for this segment to maximize approvals and approved volume while keeping costs and latency in check?

This PoC models that decision using a lightweight approach that is easy to inspect.

## What this PoC does

- **Baseline KPIs** on the currently active dataset
- **Upload a CSV dataset** to replace the sample data
- **Train a segment-level routing policy** (country + card_brand)
- **Route a new payment attempt** using the learned policy
- **Simulate smart routing KPIs** on the dataset (for demo-style uplift comparison)
- **Generate a report** (HTML, PDF, JSON) with baseline vs smart deltas and top segments

## What this PoC does not do

- No PCI scope. Do not put real PAN, CVC, or sensitive payment data in the CSV.
- No real-time health checks, failover, retries, 3DS friction modeling, fraud, or chargeback risk.
- No online experimentation framework (A/B tests). Simulation here is illustrative.

## Core workflow

1. Start the API.
2. Call **`GET /baseline`** to capture baseline KPIs on the current dataset.
3. (Optional) Upload your CSV via **`POST /upload`**.
4. Train the router via **`POST /train-smart`**.
5. Ask for a decision via **`POST /route`**.
6. Compare outcomes via **`GET /simulate-smart`**.
7. Export the report via **`GET /report?format=html|pdf|json`**.

FastAPI docs are available at **`/docs`**.

## Run locally

### 1) Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### 2) Start the server

```bash
nohup uvicorn main:app --host 0.0.0.0 --port 8000 &

```

Open:

-   API docs: `http://localhost:8000/docs`
    
-   HTML report: `http://localhost:8000/report?format=html`
    

## Run with Docker

### Build

```bash
docker build -t smart-payment-router-backend .

```

### Run

```bash
docker run --rm -p 8000:8000 smart-payment-router-backend

```

Then open `http://localhost:8000/docs`.

## API endpoints and examples

### 1) Baseline KPIs

```bash
curl http://localhost:8000/baseline

```

**Baseline metrics definition**

The API computes:

-   `approval_rate`: approved_count / total_transactions
    
-   `approved_volume`: sum(amount) for approved transactions
    
-   `total_attempted_volume`: sum(amount) for all transactions
    
-   `avg_fee`: mean(fee(provider, amount)) across all transactions
    
-   `p95_latency`: 95th percentile of `latency_ms`
    
-   `total_transactions`: number of rows in the dataset
    

Provider fee is currently modeled as:

`fee = fixed + pct * amount`

Fees are configured in `main.py` under `PROVIDER_FEES`.

### 2) Upload a dataset (CSV)

```bash
curl -F "file=@data/sample_transactions.csv" http://localhost:8000/upload

```

This writes the CSV to `data/uploaded_transactions.csv`, switches the active dataset to that file, and clears any previously learned routing config.

### 3) Train smart routing

```bash
curl -X POST http://localhost:8000/train-smart

```

Training output includes a list of segments and each segment's provider order.

### 4) Route a payment attempt

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"country":"FR","card_brand":"visa","amount":120.0,"device":"mobile"}'

```

Response example:

```json
{
  "provider_order": ["B", "A", "C"],
  "primary_provider": "B",
  "segment": "FR|visa",
  "reason": "Segment FR|visa: primary provider 'B' maximizes (predicted approval × amount − fees) on your dataset."
}

```

### 5) Simulate smart routing KPIs

```bash
curl http://localhost:8000/simulate-smart

```

**How simulation works**

-   The policy selects a primary provider for each segment (country + card_brand).
    
-   For each historical row, we replace its provider with the segment's primary provider.
    
-   Approval is simulated using the empirical approval rate observed for that (country, card_brand, provider) triple in the dataset.
    
-   A fixed random seed is used for reproducibility.
    

This is intentionally simple and is meant to be demo-friendly, not statistically rigorous.

### 6) Get the learned routing config

```bash
curl http://localhost:8000/smart-config

```

The config is stored at `data/smart_routing.json`.

### 7) Export the report

-   HTML: `http://localhost:8000/report?format=html`
    
-   PDF: `http://localhost:8000/report?format=pdf`
    
-   JSON: `http://localhost:8000/report?format=json`
    

The report includes baseline KPIs, smart KPIs, deltas, and a top list of learned segments.

## How training and routing work

### Segmentation

Segments are built as:

-   `segment_id = "{country}|{card_brand}"`
    

### Model

A logistic regression is trained to predict success (`approved` vs `declined`) using:

-   country (encoded)
    
-   card_brand (encoded)
    
-   device (encoded)
    
-   amount
    
-   provider (encoded)
    

### Provider scoring per segment

For each segment, each provider is scored as:

`score = predicted_approval_probability * amount - fee(provider, amount)`

Providers are sorted descending by score to create `routing_order`.

## Dataset format

A sample dataset is provided at `data/sample_transactions.csv`.

Required columns:

-   `transaction_id`
    
-   `timestamp`
    
-   `country`
    
-   `card_brand`
    
-   `amount`
    
-   `currency` (informational)
    
-   `device`
    
-   `provider`
    
-   `status` (`approved` or `declined`)
    
-   `error_code` (informational)
    
-   `latency_ms`
    

## Project layout

```
.
├─ main.py
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .dockerignore
├─ README.md
└─ data/
   └─ sample_transactions.csv

```
