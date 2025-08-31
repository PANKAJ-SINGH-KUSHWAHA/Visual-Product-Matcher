# Visual Product Matcher — Backend (FastAPI)

## Quick start (local)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

The first run will download model weights and build an embeddings index by fetching images from `products.csv` URLs.

## API
- `GET /health` → `{ ok: true, count: <int> }`
- `GET /products` → list of products
- `POST /search` (multipart/form-data)
  - fields: `file` (image) *or* `image_url` (string)
  - optional: `k` (int, default 10), `min_score` (float, default 0.0)
  - returns: `{ results: [ { id, name, category, image_url, score } ] }`

## Deploy (Render.com free tier)
1. Create a new **Web Service** from this folder's repo.
2. Runtime: Python 3.10+
3. Build command: `pip install -r backend/requirements.txt`
4. Start command: `uvicorn backend.main:app --host 0.0.0.0 --port 10000`
5. Add environment variable: `PORT=10000`
6. Once live, note the base URL (e.g., `https://your-api.onrender.com`).

## Notes
- Uses **ResNet50** embeddings + **FAISS** (cosine similarity).
- Images are fetched via URL (no local dataset needed). Replace `products.csv` with your own catalog.
