# guarantorLens_mission_capstone_BE

Backend for the guarantorLens mission capstone project (FastAPI/Flask-style app).

## Requirements

- Python 3.10+
- A virtual environment (recommended)

This repository includes a `requirements.txt` with the minimal runtime dependencies for the API.

Key packages:

- `fastapi` — the web framework
- `uvicorn[standard]` — ASGI server for development
- `SQLAlchemy` — ORM and database access
- `psycopg2-binary` — Postgres driver (used when connecting to Postgres)
- `python-dotenv` — load environment variables from a `.env` file

## Setup

Create and activate a virtual environment, then install dependencies from `requirements.txt`:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

Run the app with your chosen runner. Example using Python directly:

```bash
# from project root
export FLASK_APP=app.main
flask run
```

Or with `uvicorn` for FastAPI:

```bash
uvicorn app.main:app --reload
```

## Environment variables (.env)

Use a `.env` file if your application requires secrets or configuration that vary by environment, for example:

- Database URL (e.g. `DATABASE_URL`)
- Secret keys (e.g. `SECRET_KEY`)
- API credentials

A `.env` is OPTIONAL for purely local, hard-coded configs, but RECOMMENDED if you store credentials or different settings per environment. This repository already ignores `.env` in `.gitignore`.

Example `.env`:

```
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
SECRET_KEY=change-me
DEBUG=True
```

Do NOT commit real secrets. Keep `.env` out of version control (already added to `.gitignore`).

## Notes

- This project uses FastAPI; run it with `uvicorn app.main:app --reload`.
- If you deploy to a platform that provides a different `DATABASE_URL` format (e.g. `postgres://`), the code in `app/db.py` handles conversion for SQLAlchemy.

If you want, I can pin exact package versions and add a `pip freeze > requirements.txt` output for reproducible installs.

## Deploying to Render

Render detects Python services and installs dependencies from `requirements.txt`.

For this repo, the only deployment-side file you need is `gunicorn.conf.py` at the project root. It forces Gunicorn to use the Uvicorn worker, which FastAPI requires.

Recommended Render setup:

1. Create a Render Web Service and connect this repository.
2. Use `gunicorn app:app` as the start command if Render asks for one. The repo-level `gunicorn.conf.py` will still force the correct ASGI worker.
3. Set environment variables such as `DATABASE_URL` and `SECRET_KEY` in Render's dashboard.
4. Keep `requirements.txt` at the repo root so Render can install the dependencies.

Notes:
- `gunicorn.conf.py` is the repo-side fix for the sync-worker error.
- `app/__init__.py` exports the FastAPI app so `gunicorn app:app` resolves correctly.
- If you want infrastructure-as-code later, you can add a new `render.yaml`, but it is not required for this setup.

