# Task 1: Data Collection and Preprocessing
import json
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

file_path = '/Users/jack/Documents/GitHub/MakeSense/News_crawler/newsTitle.json'  # Specify the path to the JSON file containing news titles
with open(file_path, 'r', encoding='utf-8') as file:
    news_data = json.load(file)

news_titles = [title.strip() for title in news_data if title.strip()]

# Perform data preprocessing tasks (cleaning, tokenization, lowercase conversion, etc.)

# Task 2: Text Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1, 5))
tfidf_vectors = tfidf_vectorizer.fit_transform(news_titles)

def determine_optimal_clusters(vectors):
    inertias = []
    for n_clusters in range(2, 300):  # Adjust the range as needed
        kmeans_model = KMeans(n_clusters=n_clusters)
        kmeans_model.fit(vectors)
        inertias.append(kmeans_model.inertia_)
    plt.plot(range(2, 300), inertias)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('KMeans Inertia over Different Cluster Counts')
    plt.show()

# determine_optimal_clusters(tfidf_vectors)

# Task 3: Apply K-means Clustering
from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=300)  # Specify the desired number of clusters
kmeans_labels = kmeans_model.fit_predict(tfidf_vectors)

# Task 4: Keyword Extraction and File Saving
from keybert import KeyBERT

keybert_model = KeyBERT('distilbert-base-nli-mean-tokens')  # Specify the model to use for keyword extraction

def extract_keywords_for_clusters(labels, texts, num_keywords=5):
    cluster_keywords = {}
    for label in set(labels):
        # Filter texts by cluster label
        cluster_texts = [text for text, lbl in zip(texts, labels) if lbl == label]
        # Concatenate all texts in the cluster into one string
        combined_texts = ' '.join(cluster_texts)
        # Extract keywords
        keywords = keybert_model.extract_keywords(combined_texts, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_keywords)
        # Store keywords for the cluster
        cluster_keywords[label] = [word for word, _ in keywords]
    return cluster_keywords

cluster_keywords = extract_keywords_for_clusters(kmeans_labels, news_titles)

def save_results_with_keywords_to_txt(labels, texts, keywords, filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        for label in set(labels):
            # Write cluster number
            file.write(f'Cluster {label}:\n')
            # Write keywords for the cluster
            file.write(f"Keywords: {', '.join(keywords[label])}\n")
            # Write titles in the cluster
            cluster_indices = [i for i, lbl in enumerate(labels) if lbl == label]
            for index in cluster_indices:
                file.write(f"{texts[index]}\n")
            file.write("\n")

output_path = '/Users/jack/Documents/GitHub/MakeSense/AI_model/cluster_results.txt'  # Specify the path to save the clustering results
save_results_with_keywords_to_txt(kmeans_labels, news_titles, cluster_keywords, output_path)

# Task 5: Generate Sentences Based on Keywords
import random
from sklearn.metrics.pairwise import cosine_similarity

def generate_sentences(keywords):
    generated_sentences = []
    # Example: Randomly select a cluster
    # Randomly select one keyword from the cluster
    random_cluster = random.choice(list(keywords.keys()))
    random_keyword = keywords[random_cluster]
    sentence = f"{random_keyword} 해당 뉴스 키워드를 참고하여 작문하시오."
    generated_sentences.append(sentence)
    return generated_sentences

keywords = cluster_keywords

generated_sentences = generate_sentences(keywords)
print(generated_sentences)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)  # reduce to 2 dimensions for visualization
reduced_vectors = pca.fit_transform(tfidf_vectors.toarray())

# Convert cluster labels to integers for color mapping
kmeans_labels_int = np.array(kmeans_labels, dtype=np.int64)

# Plot the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], hue=kmeans_labels_int, palette='viridis', legend='full')
plt.title('2D PCA of KMeans Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Plot the cluster centroids
centroids = pca.transform(kmeans_model.cluster_centers_)
plt.figure(figsize=(12, 8))
sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], hue=kmeans_labels_int, palette='viridis', alpha=0.5)
sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], s=100, color='black', marker='X', label='Centroids')
plt.title('2D PCA of KMeans Clusters with Centroids')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()


from sklearn.metrics import silhouette_score

# Assuming 'tfidf_vectors' is your vectorized data and 'kmeans_labels' are the labels from KMeans
# You need to ensure these variables are defined and contain your data and labels

# Calculate silhouette score
silhouette_avg = silhouette_score(tfidf_vectors, kmeans_labels)
silhouette_percentage = (silhouette_avg + 1) / 2 * 100
print("silhouette score", silhouette_percentage, "%")
# Additional code for visualization, further tasks, etc.