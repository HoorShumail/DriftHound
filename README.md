# 🐕‍🦺 DriftHound
### AI-Powered ML Model Monitoring with Statistical Drift Detection & Automated Diagnostics

DriftHound is a full-stack monitoring tool that catches ML models silently failing in production. It compares incoming data against a reference baseline using three independent statistical tests, flags exactly which features have drifted, and uses an AI diagnostic agent to explain what happened and what to do about it — in plain English, not raw p-values.

It helps users and developers:

- 🎯 Catch model drift before it becomes a business problem
- 📊 Run three independent statistical tests per feature, not just one
- 🧠 Get an AI-generated root cause analysis and action plan
- 📤 Analyze real data by uploading their own baseline/current CSVs
- 🚀 Explore a live, deployed example of a monitoring + diagnostic pipeline

## 🚀 Features

✅ Multi-Test Drift Detection (KS Test, PSI, Wasserstein Distance)

✅ Any-Test-Triggers Status Logic (favors catching real drift over reducing noise)

✅ AI Diagnostic Agent powered by GPT-4o-mini

✅ CSV Upload for Custom Baseline/Current Data

✅ One-Click Synthetic Pipeline for Instant Demo Data

✅ Live Distribution Charts (Baseline vs. Current, per feature)

✅ Model Performance Metrics Dashboard (Accuracy, Precision, Recall, F1, ROC AUC)

✅ Next.js Dashboard with Real-Time Status Banner

## 🏗️ Project Architecture

```
Baseline Data + Current Data
          │
          ▼
     DriftMonitor
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
 KS Test  PSI  Wasserstein
    │     │     │
    └─────┼─────┘
          ▼
   Feature Drift Report (JSON)
          │
          ▼
   AI Diagnostic Agent (GPT-4o-mini)
          │
    ┌─────┴─────┐
    ▼           ▼
Root Cause   Recommended
 Analysis      Actions
    │           │
    └─────┬─────┘
          ▼
  FastAPI Backend ──► Next.js Dashboard
```

## 🛠️ Tech Stack

- Python
- FastAPI
- scikit-learn
- SciPy
- pandas / PyArrow
- OpenAI (GPT-4o-mini)
- Next.js
- React
- Tailwind CSS
- Recharts
- react-markdown

## 📂 Project Structure

```
DriftHound/
│
├── api/
│   └── server.py              # FastAPI app — all routes, CORS config
│
├── agent/
│   ├── client.py               # OpenAI client setup
│   └── diagnostician.py        # Drift → diagnosis logic
│
├── core/
│   └── drift_detectors.py      # KS, PSI, Wasserstein implementations
│
├── drift/
│   ├── monitor.py               # Orchestrates the full drift check
│   └── alerts.py                # Alert formatting and thresholds
│
├── data/
│   └── generators/
│       └── synthetic_data.py    # Synthetic baseline + drift injection
│
├── models/
│   ├── baseline.py              # Baseline training pipeline
│   ├── trainer.py
│   └── evaluator.py             # Metrics computation + comparison
│
├── config/
│   └── settings.py              # Paths, feature columns, constants
│
├── frontend/
│   └── src/
│       ├── app/
│       │   └── page.tsx         # Main dashboard page
│       └── components/
│           ├── upload-panel.tsx
│           ├── drift-table.tsx
│           ├── distribution-chart.tsx
│           ├── metrics-cards.tsx
│           ├── diagnosis-panel.tsx
│           ├── status-banner.tsx
│           └── run-pipeline-button.tsx
│
├── tests/
├── run_pipeline.py               # Standalone CLI entry point
├── main.py                       # FastAPI Cloud entrypoint
└── requirements.txt
```

## 🎯 How It Works

**1️⃣ Get Baseline and Current Data**

Either click **Run Pipeline** to generate synthetic baseline and current datasets instantly, or upload your own `baseline.csv` and `current.csv` through the dashboard.

**2️⃣ System Processing**

The system:
- Runs KS Test, PSI, and Wasserstein Distance on every feature
- Compares each result against the reference baseline
- Flags a feature as **Drifted** if any single test detects a shift
- Computes baseline model performance metrics (Accuracy, Precision, Recall, F1, ROC AUC)

**3️⃣ AI Diagnosis**

The Diagnostic Agent:
- Reads the full drift report
- Generates a **Drift Summary** (what drifted, severity, impact)
- Produces a ranked **Root Cause Analysis**
- Returns a prioritized **Recommended Actions** checklist
- Adds a **Risk Assessment** for what happens if left unaddressed

**4️⃣ System Displays**

- 📊 Model Performance metrics cards
- 📋 Feature Drift Analysis table (per-test breakdown + status)
- 📈 Distribution Overview charts (baseline vs. current, per feature)
- 🤖 AI Diagnostics panel (full structured explanation)

## 📊 Example Usage

**Input**

```
Upload baseline.csv and current.csv, then click Analyze My Data.
```

**Output**

```
📋 2 of 6 features flagged as Drifted (feature_1, feature_3)
📈 Distribution charts update to show baseline vs. current for each feature
🧠 AI Diagnosis generated: drift summary, root cause analysis,
   recommended actions, and risk assessment
✅ Dashboard refreshes with real data instead of synthetic samples
```

## 💡 Future Improvements

- Configurable Feature Schemas (beyond the current fixed columns)
- Seeded/Reproducible Synthetic Runs
- Authentication for Multi-User Deployments
- Docker Support for Local + Production Parity
- Historical Drift Tracking Over Time
- Slack/Email Alerting Integration
- Support for Additional Drift Detection Methods
- Multi-Model Comparison Dashboard

## 🔗 Links

- **Live demo:** https://drift-hound.vercel.app
- **API docs:** https://drifthound-baf0314d.fastapicloud.dev/docs
- **GitHub:** https://github.com/HoorShumail/DriftHound

## 🧑‍💻 Author

**Hoor Shumail**

AI | Machine Learning | Agentic AI | Model Monitoring | Multi-Agent Systems

- GitHub: https://github.com/HoorShumail
- LinkedIn: https://www.linkedin.com/in/hoor-shumail-a3a076326/

## 📜 License

This project is developed for educational and portfolio purposes.
