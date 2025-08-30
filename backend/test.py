import pickle
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import os
from tqdm import tqdm

# --- Load CSV mapping: filename -> URL ---
df = pd.read_csv('images.csv')  # columns: 'filename', 'link'
filename_to_url = dict(zip(df['filename'], df['link']))

# --- Load or extract embeddings ---
if os.path.exists('embeddings.pkl') and os.path.exists('urls.pkl'):
    print("Loading embeddings and filenames...")
    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('urls.pkl', 'rb'))
else:
    print("Extracting features from images folder...")
    # Load ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

    valid_extensions = ('.jpg', '.jpeg', '.png')
    filenames = [f for f in os.listdir('image') if f.lower().endswith(valid_extensions)]
    
    feature_list = []
    urls = []

    for file in tqdm(filenames, desc="Extracting features"):
        img_path = os.path.join('image', file)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img, verbose=0).flatten()
        normalized_result = result / norm(result)
        feature_list.append(normalized_result)

        urls.append(filename_to_url.get(file, None))  # Save URL for each image

    pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
    pickle.dump(urls, open('urls.pkl', 'wb'))
    filenames = urls  # For display, use URLs

# --- Load query image ---
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

query_img_path = 'sample/1.jpeg'
img = image.load_img(query_img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# --- Find nearest neighbors ---
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)
distances, indices = neighbors.kneighbors([normalized_result])

# --- Display results from URLs ---
for idx in indices[0]:
    url = filenames[idx]  # filenames.pkl now stores URLs

    if url:
        print("Displaying URL:", url)
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img_np = np.array(img)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img_cv, (512, 512))
            cv2.imshow("Result", img_resized)
            cv2.waitKey(0)
        except Exception as e:
            print(f"Failed to load {url}: {e}")
    else:
        print("No URL found for this image.")

cv2.destroyAllWindows()
