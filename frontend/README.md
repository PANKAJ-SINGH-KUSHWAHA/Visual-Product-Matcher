# Visual Product Matcher — Frontend (React + Vite + Tailwind)

## Quick start (local)
```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env.local
npm run dev
```
Then open the printed localhost URL.

## Deploy (Vercel/Netlify)
- Set environment variable `VITE_API_URL` to your deployed backend URL (e.g., `https://your-api.onrender.com`).
- Build command: `npm run build`
- Output directory: `dist`
