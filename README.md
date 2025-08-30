# Visual Product Matcher (Full Stack)

A minimal, production-quality **Visual Product Matcher** built with:
- **Frontend**: React (Vite) + Tailwind
- **Backend**: FastAPI + ResNet50 (torchvision) + FAISS (cosine similarity)
- **Data**: `backend/products.csv` with 50 sample items using picsum URLs

---

## Run locally

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
`GET /health` should return a JSON with a count.

### Frontend
```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env.local
npm run dev
```

---

## Deploy
- **Backend** → Render.com / Railway.app (free)
  - Build: `pip install -r backend/requirements.txt`
  - Start: `uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}`
- **Frontend** → Vercel / Netlify
  - Env: `VITE_API_URL` set to your backend URL
  - Build: `npm run build`
  - Output: `dist`

---

## 200-word approach (paste into your submission)
We built a lightweight visual search engine. The backend uses a pre-trained ResNet50 to convert images into 2048‑dim embeddings and normalizes them so cosine similarity reflects visual closeness. Product data comes from a simple CSV with image URLs; on first run, the API downloads each image, computes embeddings, and builds a FAISS inner‑product index for fast k‑NN retrieval. The API exposes `/search` to accept either an uploaded image or an image URL, returning ranked matches with scores; `/products` lists the catalog. FastAPI provides clean, typed endpoints and CORS for the React client. The frontend (Vite + React + Tailwind) offers file upload or URL input, shows a live preview, adjustable Top‑K and min‑score filters, loading states, and error handling. Cards show product image, name, category, and score. The UI is mobile‑first and responsive. For deployment, the backend runs on Render/Railway free tiers, and the frontend on Vercel/Netlify with an environment variable for the API base URL. This design is simple, production‑ready, and easy to extend (replace CSV with a DB, add categories/filters, or swap the embedding model).
