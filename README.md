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
  models.py          # User, Application, Recommendation, AuditLog tables
  schema.py          # Pydantic request/response models
  scoring.py         # model serving: score, SHAP, flags, brief, advisor, what-if
  risk.py            # /assess-risk, /assess/suggest-guarantors
  members.py         # member list, search, detail
  network_data.py    # loan/edge tables, ego networks, portfolio rollups
  insights.py        # overview, weak-links, contagion, early-warning, communities
  applications.py    # applications, escalate, recommendations, audit log (role-gated)
  admin.py           # replace model/artifacts, clear applications
  artifacts/         # guarantorlens_serving.joblib, members.json, loans.json
tests/               # pytest suite: unit, validation, integration, functional,
                     #   acceptance, fallback, audit-trail
requirements.txt     # runtime deps (note the scikit-learn pin)
requirements-dev.txt # test-only deps (pytest, httpx, pytest-cov)
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
- **Applications:** `POST /applications`, `GET /applications` (+ `?escalated=true` review queue), `GET /applications/{id}`, `POST /applications/{id}/escalate`, `POST /applications/{id}/recommendations`, `GET /applications/{id}/audit` (append-only decision/override log)
- **Admin:** replace the model/artifacts, clear applications

### Roles & permissions

| Role | Can do |
| --- | --- |
| `loan_officer` | Assess loans, create and escalate applications (cannot record a recommendation) |
| `credit_manager` | Everything an officer can, plus review the escalation queue and record recommendations |
| `admin` | Manager rights plus replacing the deployed model/artifacts |

The recommendation endpoint returns **403** for a non-manager, so an officer cannot approve their own case. This is enforced server-side, not just hidden in the UI.

### Audit log

Every decision and override is written to an **append-only `audit_log` table** (`AuditLog` in `models.py`): one immutable row per action (`assess` / `escalate` / `recommend`), attributed to the acting user with name, role, a JSON detail snapshot, and a timestamp. Nothing updates or deletes these rows. A credit manager (or the owning officer) reads an application's trail at `GET /applications/{id}/audit`; there is no create/edit/delete route for the log.

---

## Tests

The suite is **44 tests across 7 categories** (unit, validation, integration, functional, acceptance, fallback, audit-trail). It needs no running server or external database: `tests/conftest.py` points the app at a throwaway SQLite file and exposes a `TestClient` plus ready-made officer and manager tokens.

### How to run

```bash
# 1. install the test-only dependencies (pytest, httpx, pytest-cov)
pip install -r requirements-dev.txt

# 2. run the whole suite FROM THE REPO ROOT
python -m pytest

# 3. with a coverage report (overall ~64%; scoring/decision path is highest)
python -m pytest --cov=app --cov-report=term-missing
```

> Run it as `python -m pytest`, **not** bare `pytest`. The `python -m` form puts the repo root on the path so the `app` package imports; the bare command fails with `ModuleNotFoundError: No module named 'app'`.

More ways to run:

```bash
python -m pytest -v                                  # verbose: list every test name
python -m pytest tests/test_validation.py            # one category (file)
python -m pytest tests/test_scoring_unit.py -v        # one file, verbose
python -m pytest -k "recommendation"                 # only tests matching a keyword
python -m pytest tests/test_scoring_unit.py::test_band_boundaries   # a single test
python -m pytest --collect-only -q                   # list tests without running
```

### What each category locks in

| Category | File | Tests | Covers |
| --- | --- | --- | --- |
| Unit | `tests/test_scoring_unit.py` | 9 | Band boundaries; display score aligns with the band and is monotonic in probability; two written-off guarantors escalate the band to High; metamorphic sanity (more savings never raises risk, a bigger loan never lowers it); the `assess()` response shape. |
| Validation | `tests/test_validation.py` | 5 | Malformed / out-of-range requests are rejected with clear errors; required fields (borrower, at least one guarantor) are enforced. |
| Integration | `tests/test_api_integration.py` | 6 | Health; auth required; `/assess-risk` returns a valid band and score; a well-covered loan scores no higher than a thin one; **an officer is blocked (403) from recording a recommendation** while a manager succeeds. |
| Functional | `tests/test_functional_cases.py` | 6 | End-to-end scenarios across the risk spectrum produce the expected bands and flags (e.g. a loan backed by a written-off member is flagged). |
| Acceptance | `tests/test_acceptance.py` | 3 | User-story criteria: officer proposes and escalates, manager reviews and records a recommendation, and every assessment returns a plain-language explanation. |
| Fallback | `tests/test_fallback.py` | 5 | Model failure and graceful degradation: a missing bundle, an empty model, and a model that raises while scoring all fall back to the rule-based heuristic; the API still answers and tags the response `source="heuristic"`. |
| Audit trail | `tests/test_audit_trail.py` | 10 | Every decision and override writes an attributed, append-only `audit_log` row (assess / escalate / recommend), captures what-if overrides, stays immutable (no edit/delete route), and enforces separation of duties. |

Because the tests import the **real model bundle**, they double as a **deployment smoke test**: if the scikit-learn version ever drifts and the model can't unpickle, the assess tests fail instead of the API silently dropping to the rule-based fallback.

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
