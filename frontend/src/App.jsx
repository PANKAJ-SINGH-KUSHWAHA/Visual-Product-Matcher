import { useState, useMemo } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000'

export default function App() {
  const [file, setFile] = useState(null)
  const [imageUrl, setImageUrl] = useState('')
  const [preview, setPreview] = useState(null)
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [k, setK] = useState(5)

  const handleFile = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    setImageUrl('')
    setPreview(URL.createObjectURL(f))
  }

  const handleSearch = async () => {
    try {
      setError('')
      setLoading(true)
      setResults([])

      const form = new FormData()
      if (file) form.append('file', file)
      if (imageUrl && !file) form.append('image_url', imageUrl)
      form.append('k', String(k))

      const { data } = await axios.post(`${API_URL}/search`, form, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setResults(data.results || [])
    } catch (err) {
      if (err.response?.status === 404) {
        setError('No similar product found.')
      } else {
        setError(err?.response?.data?.detail || 'Search failed')
      }
    } finally {
      setLoading(false)
    }
  }

  const disabled = useMemo(() => !file && !imageUrl, [file, imageUrl])

  return (
    <div className="min-h-screen bg-gradient-to-b from-pink-50 to-white text-gray-900">
      {/* Header */}
      <header className="border-b bg-white/70 backdrop-blur-md shadow-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row md:items-center md:justify-between p-4 gap-1">
          <h1 className="text-2xl font-bold tracking-tight">Visual Product Matcher</h1>
          <p className="text-sm text-gray-600">
            Built by <span className="font-medium">Pankaj Singh Kushwaha</span> — 
            <a 
              href="mailto:kushwahapankaj793@gmail.com"
              className="text-pink-600 hover:underline"
            >
              kushwahapankaj793@gmail.com
            </a>
          </p>
        </div>
      </header>


      <main className="max-w-6xl mx-auto p-6 grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Left Controls */}
        <div className="space-y-6">
          <div className="p-5 bg-white rounded-2xl shadow-md">
            <label className="block text-sm font-semibold mb-2">Upload image</label>
            <input type="file" accept="image/*" onChange={handleFile} />
          </div>

          <div className="p-5 bg-white rounded-2xl shadow-md">
            <label className="block text-sm font-semibold mb-2">or Paste Image URL</label>
            <input
              type="url"
              placeholder="https://..."
              value={imageUrl}
              onChange={(e) => { setImageUrl(e.target.value); setFile(null); setPreview(e.target.value || null); }}
              className="w-full border rounded-xl px-3 py-2"
            />
          </div>

          <div className="p-5 bg-white rounded-2xl shadow-md space-y-4">
            <div>
              <label className="block text-sm font-semibold mb-1">Top K results: {k}</label>
              <input type="range" min="3" max="8" value={k}
                onChange={(e) => setK(parseInt(e.target.value))} className="w-full accent-pink-500" />
            </div>
            <button
              onClick={handleSearch}
              disabled={disabled || loading}
              className="w-full py-3 rounded-xl bg-gradient-to-r from-pink-500 to-red-500 text-white font-semibold hover:opacity-90 transition"
            >
              {loading ? 'Searching…' : 'Search'}
            </button>
            {error && <p className="text-red-600 text-sm">{error}</p>}
          </div>
        </div>

        {/* Right Preview + Results */}
        <div className="md:col-span-2 space-y-6">
          {/* Preview */}
          <div className="p-5 bg-white rounded-2xl shadow-md">
            <h2 className="text-lg font-semibold mb-2">Preview</h2>
            {preview ? (
              <img src={preview} alt="preview" className="max-h-80 rounded-xl mx-auto shadow-lg" />
            ) : (
              <div className="h-80 flex items-center justify-center text-gray-400 border-2 border-dashed rounded-xl">
                No image selected
              </div>
            )}
          </div>

          {/* Results */}
          <div className="p-5 bg-white rounded-2xl shadow-md">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold">Similar Products</h2>
              <span className="text-xs text-gray-500">{results.length} results</span>
            </div>
            {results.length === 0 ? (
              <div className="text-sm text-gray-500 text-center py-8">No results yet. Try searching!</div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-5">
                {results.map((r) => (
                  <div key={r.id} className="border rounded-2xl overflow-hidden bg-white hover:shadow-lg transition">
                    <img src={r.image_url} alt={r.name} className="w-full h-48 object-cover transform hover:scale-105 transition" />
                    <div className="p-3">
                      <div className="text-sm font-medium">{"Similar Product"}</div>
                      <div className="text-xs text-gray-500">Score: {r.score.toFixed(3)}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
