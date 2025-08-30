import pickle
import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow
import io
import os
import pandas as pd
from numpy.linalg import norm
from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# Load model
# -------------------------
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tensorflow.keras.Sequential([
    base_model,
    GlobalAveragePooling2D()
])

# -------------------------
# Load embeddings + URLs
# -------------------------
feature_list = pickle.load(open("embeddings.pkl", "rb"))
filenames = pickle.load(open("urls.pkl", "rb"))  # these are the filenames or URLs

feature_list = np.array(feature_list, dtype="float32")
index = faiss.IndexFlatL2(feature_list.shape[1])
index.add(feature_list)

# -------------------------
# Load CSV mapping: filename -> URL
# -------------------------
df = pd.read_csv("images.csv")  # columns: filename, link

# Keep only CSV rows that have embeddings
filenames_only = set([os.path.basename(f) for f in filenames])
df = df[df['filename'].isin(filenames_only)].reset_index(drop=True)

filename_to_url = dict(zip(df['filename'], df['link']))

# -------------------------
# Map filename -> ID (use filename string itself)
# -------------------------
filename_to_id = {os.path.basename(f): os.path.basename(f) for f in filenames}

# -------------------------
# Helper: extract features
# -------------------------
def extract_features_from_bytes(file_bytes):
    img = image.load_img(io.BytesIO(file_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / norm(result)
    return normalized_result.astype("float32")

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow your frontend
    allow_credentials=True,
    allow_methods=["*"],    # allow all HTTP methods
    allow_headers=["*"],    # allow all headers
)

@app.post("/search")
async def search(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    k: int = Form(8)
):
    if file:
        contents = await file.read()
    elif image_url:
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch image URL")
        contents = response.content
    else:
        raise HTTPException(status_code=400, detail="No file or image URL provided")

    # Extract features
    query_vector = extract_features_from_bytes(contents)

    # Search FAISS
    distances, indices = index.search(np.array([query_vector]), k=k)
    if indices is None or len(indices[0]) == 0:
        return {"results": []}

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        img_url = filenames[idx]  # this should be a proper HTTP URL
        score = float(1 / (1 + dist))
        results.append({
            "id": str(idx),       # use index as id
            "score": score,
            "image_url": img_url
        })

    # Sort by score descending
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return {"results": results[:k]}