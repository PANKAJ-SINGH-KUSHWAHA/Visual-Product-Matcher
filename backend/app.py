import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import pandas as pd
from tqdm import tqdm
import pickle

# Load CSV mapping: filename -> URL
df = pd.read_csv('images.csv')  # columns: 'filename', 'link'
filename_to_url = dict(zip(df['filename'], df['link']))

# Load ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D()
])

# Feature extraction function
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / norm(result)
    return normalized_result  

# Collect all valid image files in the 'images' folder
valid_extensions = ('.jpg', '.jpeg', '.png')
filenames = [f for f in os.listdir('images') if f.lower().endswith(valid_extensions)]

# Extract features and store URLs
feature_list = []
urls_list = []

for file in tqdm(filenames, desc="Extracting features"):
    img_path = os.path.join('images', file)
    features = extract_features(img_path, model)
    feature_list.append(features)
    
    # Get URL from CSV mapping
    url = filename_to_url.get(file, None)
    if url is None:
        print(f"Warning: No URL found for {file}")
    urls_list.append(url)

# Save embeddings and URLs
pickle.dump(np.array(feature_list), open('embeddings.pkl', 'wb'))
pickle.dump(urls_list, open('urls.pkl', 'wb'))

print(f"Processed {len(feature_list)} images. URLs saved for {len([u for u in urls_list if u])} images.")
