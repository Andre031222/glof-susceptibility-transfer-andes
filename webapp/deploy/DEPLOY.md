# Deploy — glof.ginit.dev

FastAPI + PostgreSQL + nginx on your VPS. Frontend is Tailwind (CDN, no build step).

## 0. DNS
Add an **A record**: `glof` -> your VPS IP (in the ginit.dev zone). Wait for propagation.

## 1. PostgreSQL (via pgAdmin4 or psql)
```sql
CREATE USER glof WITH PASSWORD 'STRONG_PASSWORD';
CREATE DATABASE glof OWNER glof;
```
Tables are created automatically on first API start (SQLAlchemy), or load `backend/schema.sql` manually.

## 2. Code + venv
```bash
sudo mkdir -p /opt/glof-andes && sudo chown $USER /opt/glof-andes
cp -r webapp/backend webapp/frontend /opt/glof-andes/
python3 -m venv /opt/glof-andes/venv
/opt/glof-andes/venv/bin/pip install -r /opt/glof-andes/backend/requirements.txt
cp /opt/glof-andes/backend/.env.example /opt/glof-andes/backend/.env   # edit GLOF_DB_URL
```

## 3. Seed lakes + thumbnails (run after the corrected pipeline + validation sheet exist)
```bash
mkdir -p /opt/glof-andes/thumbs
GLOF_DB_URL=... GLOF_THUMBS=/opt/glof-andes/thumbs \
  /opt/glof-andes/venv/bin/python /opt/glof-andes/backend/seed.py
```

## 4. Service
```bash
sudo cp webapp/deploy/glof-api.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now glof-api
sudo systemctl status glof-api          # check it is running on 127.0.0.1:8077
```

## 5. nginx + HTTPS
```bash
sudo cp webapp/deploy/nginx-glof.conf /etc/nginx/sites-available/glof.ginit.dev
sudo ln -s /etc/nginx/sites-available/glof.ginit.dev /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d glof.ginit.dev   # automatic TLS
```

Open https://glof.ginit.dev — identify with a name and start validating.

## Update after re-running the pipeline
Re-run `seed.py` (it replaces lakes + thumbs). Reviewer labels are preserved
unless a `lake_key` disappears (its labels cascade-delete).
