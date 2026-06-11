# guarantorLens_mission_capstone_BE

Backend for the guarantorLens mission capstone project — a [FastAPI](https://fastapi.tiangolo.com) application.

🔗 **Live API docs (Swagger UI):** https://guarantorlens-mission-capstone-be.onrender.com/docs

---

## Requirements

- Python 3.10+
- A virtual environment (recommended)

This repository includes a `requirements.txt` with the minimal runtime dependencies for the API.


## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Run

Start the development server with Uvicorn:

```bash
# from the project root
uvicorn app.main:app --reload
```

The API will be available at **http://127.0.0.1:8000**, and the interactive docs (Swagger UI) at **http://127.0.0.1:8000/docs**.

---

## Environment Variables (`.env`)

Use a `.env` file for secrets and configuration

- Database URL (e.g. `DATABASE_URL`)


**Example `.env`:**

```
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
SECRET_KEY=change-me
DEBUG=True
```

## Deploying to Render

Render auto-detects Python services and installs dependencies from `requirements.txt`.

For this repo, the only deployment-side file needed is `gunicorn.conf.py` at the project root. It forces Gunicorn to use the Uvicorn worker, which FastAPI requires.

**Recommended Render setup:**

1. Create a Render **Web Service** and connect this repository.
2. Use `gunicorn app:app` as the start command if Render asks for one. The repo-level `gunicorn.conf.py` will still force the correct ASGI worker.
3. Set environment variables such as `DATABASE_URL` and `SECRET_KEY` in Render's dashboard.
4. Keep `requirements.txt` at the repo root so Render can install the dependencies.

