# Local verification — 17 October 2026

Windows/Python 3.10.9; existing packages: Streamlit 1.53.1, pandas 2.1.4, NumPy 1.26.3, scikit-learn 1.7.2, XGBoost 3.2.0, Plotly 6.5.2 and python-dotenv 1.2.1. No clean dependency installation, model training or paid API request was performed.

From the repository root:

```powershell
python -u docs/verify_local.py
```

**Result: exit 0.** This executes actual modules with new synthetic data, not mocked predictions:

| Check | Observed result |
|---|---|
| Generator, seed 42, `save_to_file=False` | 160 rows, 75 columns, 25% rule-generated churn; generator validation passed |
| Preprocessing, `save_output=False` | Train `(128, 75)`, test `(32, 75)`; matching columns, no missing training values |
| Risk-tier boundaries | 0/.2/.4/.6/.8/1 map to minimal/low/medium/high/critical/critical |
| Exposure arithmetic | $1,000 monthly, 0.5 probability, 12-month tenure gives $6,000 expected annual loss and $12,000 lifetime value at risk; illustrative formula check |
| Streamlit AppTest home | Rendered with 200 illustrative rows, no application exception |
| All five pages | Rendered with default controls and shared demo state, no application exception |

AppTest emitted Streamlit deprecation warnings for `use_container_width`; these did not fail the checks. Buttons and every widget combination were not exhaustively exercised. Temporary input data was discarded; tracked data/model artifacts were not overwritten.

## Browser evidence

From the repository root:

```powershell
python -m streamlit run dashboard/app.py --server.headless true --server.address 127.0.0.1 --server.port 8113 --browser.gatherUsageStats false
```

T3 `preview_open` reached the page at `http://127.0.0.1:8113`. Initial screenshot attempts timed out; after browser reconnection a new dedicated tab successfully captured this source edition with `preview_snapshot({save:true})`. The preview [assets/dashboard-local.png](assets/dashboard-local.png) was copied from that actual 8113 capture. It shows a separate 200-row synthetic table, not trained-model output. Displayed business values are illustrative calculations.

## Scope and outstanding work

No saved pickle/joblib was loaded for this check. No full XGBoost training, independent evaluation, SHAP computation, CUSUM lead-time validation, Gemini request or real retention outcome was measured. UI charts and synthetic rule scores demonstrate interactions only. Known integration mismatches, differing risk definitions, feature-engineering leakage and business assumptions remain documented in the [preserved guide](PROJECT_GUIDE.md).

Application source is at the repository root. All 53 tracked files under src/, config/, dashboard/ and run_demo.py matched the companion repository by SHA-256. This repository was independently executed for the checks above.
