# GuarantorLens — Backend & Model Serving

Explainable, network-aware loan-default risk **decision support** for a savings and credit cooperative (Umwalimu SACCO, Rwanda). GuarantorLens scores the default risk of a proposed loan, explains every score in plain language, and reads the **guarantor network** behind the loan. It is decision support, not automatic approval: an officer proposes, a credit manager decides.

This repository is the [FastAPI](https://fastapi.tiangolo.com) backend and the model serving layer. The React frontend lives in a separate repo.

- 🔗 **Live API docs (Swagger UI):** https://guarantorlens-mission-capstone-be.onrender.com/docs
- 🔗 **Live app (frontend):** https://guarantor-lens-mission-capstone-fe.vercel.app/
- 🔗 **Frontend repo:** https://github.com/Best-Verie/guarantorLens_mission_capstone_FE

---

## What it does

- **Risk assessment** — scores a proposed loan (0–100 + Low / Medium / High band) from the borrower, the loan terms, and the guarantors.
- **Plain-language explanation** — every assessment returns a decision brief, the top drivers behind the score (SHAP), and what those drivers mean in words.
- **Guarantor-network flags** — raises the risk band on concentrated red flags, for example a backing group that is over-committed or guarantors who have defaulted before.
- **What-if** — re-score a loan with a smaller amount, more savings, or different guarantors to see how the risk moves.
- **Fix-it advisor** — for a risky loan, suggests swapping a weak guarantor for a stronger same-branch member and shows the new score.
- **Network views** — a member's guarantee network, contagion ("if this member fails, what is exposed?"), and portfolio weak links (single points of failure).
- **Roles & review queue** — loan officers propose and escalate; credit managers review and record recommendations.

---

## Architecture

```
React + Vite + TS  ─HTTP─▶  FastAPI  ─▶  SQLAlchemy (Postgres in prod, SQLite locally)
   (Vercel)                 (Render)  └▶  Model bundle + member/loan tables (app/artifacts/)
```

- **API & auth:** FastAPI, JWT bearer tokens, SQLAlchemy models for users and applications.
- **Model serving:** a saved bundle is loaded at startup and used to score `/assess-risk`. If the bundle cannot load (for example a library-version mismatch), the API falls back to a transparent rule-based heuristic so the app never goes down.
- **Static data:** anonymized member and loan tables ship as JSON in `app/artifacts/` and power the network and insights views.

---

## The model

- **Estimator:** `CalibratedClassifierCV` (isotonic) over an imbalanced-learn `Pipeline(SimpleImputer + XGBClassifier)`, with monotone constraints on the core risk features.
- **Features (18):** 12 borrower/loan features (savings, salary, loan-to-savings, loan-to-salary, loan size, interest rate, prior history, and so on) plus 6 guarantor-network features (including the borrower's guarantee-community historical default rate).
- **Training data:** 11,015 anonymized loans across 11 branches, disbursed 2022–2023, 226 defaults (a 2.1% bad rate). Highly imbalanced, so the headline metric is PR-AUC, not accuracy.
- **Held-out performance (deployed bundle):** **PR-AUC ≈ 0.52, ROC-AUC ≈ 0.92** with borrower-grouped cross-validation and a leakage screen. That is far above the 2.1% base rate at ranking risky loans.

**Model notes (kept honest):** the guarantor network is predictive on its own and improves ranking (ROC), but because risky borrowers tend to cluster with risky guarantors (homophily), its *incremental* lift on PR-AUC is limited on this dataset. The network features are kept because they power the flags, contagion, and advisor, and because they hold up as data grows.

> ⚠️ **Version pin (important).** The model bundle is pickled with **scikit-learn 1.6.1**, xgboost 3.x, numpy 1.x. `scikit-learn` in `requirements.txt` **must stay 1.6.1** — a different version fails to unpickle the model and silently drops the API to the rule-based fallback (this was the cause of a past "rule-based fallback" incident on Render). See the comments in `requirements.txt`.

---

## Repository layout

```
app/
  main.py            # FastAPI app, router wiring, startup (create tables)
  auth.py            # register / login / me / password reset (JWT)
  security.py        # password hashing, token signing
  db.py              # SQLAlchemy engine/session (DATABASE_URL)
  models.py          # User, Application, Recommendation tables
  schema.py          # Pydantic request/response models
  scoring.py         # model serving: score, SHAP, flags, brief, advisor, what-if
  risk.py            # /assess-risk, /assess/suggest-guarantors
  members.py         # member list, search, detail
  network_data.py    # loan/edge tables, ego networks, portfolio rollups
  insights.py        # overview, weak-links, contagion, early-warning, communities
  applications.py    # applications, escalate, recommendations (role-gated)
  admin.py           # replace model/artifacts, clear applications
  artifacts/         # guarantorlens_serving.joblib, members.json, loans.json
tests/               # pytest unit + integration suite
requirements.txt     # runtime deps (note the scikit-learn pin)
requirements-dev.txt # test-only deps (pytest, httpx)
```

---

## Setup & run (local)

Requires Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```

The API runs at **http://127.0.0.1:8000**, with interactive docs at **http://127.0.0.1:8000/docs**. With no `DATABASE_URL` set it uses a local SQLite file, so it runs with zero external setup.

---

## Environment variables

Use a `.env` file (loaded via python-dotenv) or set these in your host's dashboard. All have safe local defaults except in production, where you should set `SECRET_KEY` and `MEMBER_UID_SALT`.

| Variable | Purpose | Default |
| --- | --- | --- |
| `DATABASE_URL` | DB connection (Postgres in prod) | SQLite file |
| `SECRET_KEY` | JWT signing secret | dev-insecure (change in prod) |
| `MEMBER_UID_SALT` | Salt for hashing member ids into opaque URL ids | dev default (change in prod) |
| `FRONTEND_ORIGIN` | Allowed CORS origin | `*` |
| `ACCESS_TOKEN_TTL_MIN` | Access-token lifetime (minutes) | `720` |
| `RESET_TOKEN_TTL_MIN` | Password-reset token lifetime (minutes) | `60` |
| `DEBUG` | Debug flag | `False` |

---

## API overview

Full, interactive reference at `/docs`. Main groups:

- **Auth:** `POST /auth/register`, `POST /auth/login`, `GET /auth/me`, `POST /auth/forgot-password`, `POST /auth/reset-password`
- **Assessment:** `POST /assess-risk`, `POST /assess/suggest-guarantors`
- **Members:** `GET /members` (list + search), `GET /member/{ref}`, `GET /member/{ref}/contagion`
- **Insights:** `GET /insights/overview`, `/insights/weak-links`, `/insights/early-warning`, `/insights/super-guarantors`, `/insights/communities`, `/watchlist`
- **Applications:** `POST /applications`, `GET /applications` (+ `?escalated=true` review queue), `GET /applications/{id}`, `POST /applications/{id}/escalate`, `POST /applications/{id}/recommendations`
- **Admin:** replace the model/artifacts, clear applications

### Roles & permissions

| Role | Can do |
| --- | --- |
| `loan_officer` | Assess loans, create and escalate applications (cannot record a recommendation) |
| `credit_manager` | Everything an officer can, plus review the escalation queue and record recommendations |
| `admin` | Manager rights plus replacing the deployed model/artifacts |

The recommendation endpoint returns **403** for a non-manager, so an officer cannot approve their own case. This is enforced server-side, not just hidden in the UI.

---

## Tests

Unit tests cover the scoring logic; integration tests hit the running API with auth.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

What they lock in:

- **Unit** (`tests/test_scoring_unit.py`) — band boundaries; the display score aligns with the band and is monotonic in probability; two defaulter guarantors escalate the band to High; metamorphic sanity (more savings never raises risk, a bigger loan never lowers it); the `assess()` response shape.
- **Integration** (`tests/test_api_integration.py`) — health; auth required; assess-risk returns a valid band and score; a well-covered loan scores no higher than a thin one; **an officer is blocked (403) from recording a recommendation** while a manager succeeds.

Because the tests import the real model bundle, they double as a **deployment smoke test**: if the scikit-learn version ever drifts and the model can't load, the assess tests catch it.

---

## Deploying to Render

Render auto-detects Python and installs from `requirements.txt`. A repo-level `gunicorn.conf.py` forces the Uvicorn worker that FastAPI needs.

1. Create a Render **Web Service** and connect this repository.
2. Start command: `gunicorn app.main:app` (the config file sets the ASGI worker).
3. Set `DATABASE_URL`, `SECRET_KEY`, `MEMBER_UID_SALT`, and `FRONTEND_ORIGIN` in the dashboard.
4. Keep `requirements.txt` at the repo root with `scikit-learn==1.6.1` (see the version-pin warning above). After changing dependencies, deploy with **Clear build cache**.

---

## Data & privacy

The member and loan tables shipped in `app/artifacts/` are **anonymized** (opaque client ids, no names or national ids). No personal identifiers or secrets are stored in this repository. Set `SECRET_KEY` and `MEMBER_UID_SALT` from the environment in any real deployment.
