# Task 1: Data Collection and Preprocessing
import json
import matplotlib.pyplot as plt

file_path = '/Users/jack/Documents/GitHub/MakeSense/News_crawler/newsTitle.json'  # Specify the path to the JSON file containing news titles
with open(file_path, 'r', encoding='utf-8') as file:
    news_data = json.load(file)

news_titles = [title.strip() for title in news_data if title.strip()]

# Perform data preprocessing tasks (cleaning, tokenization, lowercase conversion, etc.)

# Task 2: Text Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1, 5))
tfidf_vectors = tfidf_vectorizer.fit_transform(news_title