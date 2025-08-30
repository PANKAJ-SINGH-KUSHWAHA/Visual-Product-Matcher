import pickle
import pandas as pd

feature_list = pickle.load(open("embeddings.pkl", "rb"))
df = pd.read_csv("images.csv")

print("Number of embeddings:", len(feature_list))
print("Number of CSV rows:", len(df))