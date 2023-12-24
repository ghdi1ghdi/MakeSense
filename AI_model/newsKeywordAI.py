import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from keybert import KeyBERT

# 뉴스 타이틀을 담고 있는 JSON 파일 불러오기
file_path = '/Users/jack/Documents/GitHub/MakeSense/News_crawler/newsTitle.json'
with open(file_path, 'r', encoding='utf-8') as file:
    news_data = json.load(file)
news_titles = [title.strip() for title in news_data if title.strip()]

# Vectorize the news titles using TF-IDF
tfidf_vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1, 5))
tfidf_vectors = tfidf_vectorizer.fit_transform(news_titles)

# Perform DBSCAN clustering
dbscan_model = DBSCAN(eps=0.1, min_samples=1, metric='cosine')
dbscan_labels = dbscan_model.fit_predict(tfidf_vectors)

# Extract representative articles for each cluster
def extract_representative_articles(labels, texts):
    representative_articles = {}
    for label in set(labels):
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        cluster_texts = [texts[i] for i in indices]
        representative_articles[label] = max(cluster_texts, key=len)
    return representative_articles

representatives = extract_representative_articles(dbscan_labels, news_titles)

# ... (other parts of the code remain unchanged)

# Extract keywords using KeyBERT
keybert_model = KeyBERT('distilbert-base-nli-mean-tokens')  # specify the model to use
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

# Extract keywords for each cluster
cluster_keywords = extract_keywords_for_clusters(dbscan_labels, news_titles)

# Save clustering results and keywords to a text file
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

# Specify the path to the file where you want to save the results
output_path = '/Users/jack/Documents/GitHub/MakeSense/AI_model/representative_articles.txt'
save_results_with_keywords_to_txt(dbscan_labels, news_titles, cluster_keywords, output_path)

# ... (rest of the visualization and KMeans clustering code)


# Visualize the clusters after reducing dimensionality using PCA
def visualize_clusters(vectors, labels):
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors.toarray())
    sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], hue=labels, palette='viridis')
    plt.show()

visualize_clusters(tfidf_vectors, dbscan_labels)

# Determine the optimal number of clusters using KMeans clustering
def determine_optimal_clusters(vectors):
    inertias = []
    for n_clusters in range(2, 20):  # Adjust the range as needed
        kmeans_model = KMeans(n_clusters=n_clusters)
        kmeans_model.fit(vectors)
        inertias.append(kmeans_model.inertia_)
    plt.plot(range(2, 20), inertias)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('KMeans Inertia over Different Cluster Counts')
    plt.show()

determine_optimal_clusters(tfidf_vectors)

# ... the rest of your code for further tasks ...
