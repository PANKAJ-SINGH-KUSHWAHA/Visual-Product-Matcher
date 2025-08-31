import pickle
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, UploadFile, File, Form
import requests
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import os
import gdown
import io
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

# -------------------
# Load model + data
# -------------------
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

embeddings_path = "embeddings.pkl"
urls_path = "urls.pkl"

# Download embeddings if missing (use direct link)
if not os.path.exists(embeddings_path):
    print("Downloading embeddings...")
    gdown.download(
        "https://drive.google.com/uc?id=1UH3xFHgOIPmz70pb0QB1T7iWwddgwoLE",
        embeddings_path,
        quiet=False
    )



# Load embeddings and filenames
with open(embeddings_path, "rb") as f:
    feature_list = np.array(pickle.load(f), dtype=np.float16)
print("Embeddings loaded, length:", len(feature_list))

with open(urls_path, "rb") as f:
    filenames = pickle.load(f)

# Nearest neighbors model
neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
neighbors.fit(feature_list)

# -------------------
# FastAPI app
# -------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_features(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    expanded = np.expand_dims(img_array, axis=0)
    preprocessed = preprocess_input(expanded)
    result = model.predict(preprocessed, verbose=0).flatten()
    return result / norm(result)

@app.post("/search")
async def search(
    file: UploadFile = File(None),
    image_url: str = Form(None),
    k: int = Form(5)
):
    if file:
        contents = await file.read()
    elif image_url:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        contents = response.content
    else:
        return {"results": []}

    query_vector = extract_features(contents).reshape(1, -1)
    distances, indices = neighbors.kneighbors(query_vector, n_neighbors=k)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        results.append({
            "id": rank,
            "image_url": filenames[idx],
            "name": f"Product {rank+1}",
            "score": float(1 - dist)
        })
    return {"results": results}

# -------------------
# Run app on Render
# -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)