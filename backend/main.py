import pickle
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, UploadFile, File, Form
import requests
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
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

feature_list = pickle.load(open("embeddings.pkl", "rb"))
filenames = pickle.load(open("urls.pkl", "rb"))   # list of image URLs

neighbors = NearestNeighbors(n_neighbors=8, algorithm="brute", metric="euclidean")
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