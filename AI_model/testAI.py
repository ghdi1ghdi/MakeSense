import konlpy
from konlpy.tag import Okt
okt = Okt()
import json
import pandas as pd
import koreanize_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
import numpy as np
from sklearn.decomposition import PCA
from keybert import KeyBERT

tfidf_vectorizer = TfidfVectorizer(min_df = 3, ngram_range=(1, 5))
"""
-------------------------------뉴스 데이터 벡터화 시작--------------------------------
"""
# 1. tf-idf 임베딩
# JSON 파일 경로. 이 경로를 실제 JSON 파일 경로로 변경하세요.
file_path = '/Users/jack/Documents/GitHub/MakeSense/News_crawler/newsTitle.json'
# JSON 파일 열기 및 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    # JSON 데이터를 파이썬 객체로 로드
    today_data = json.load(file)
    
# 빈 줄 또는 공백을 제거하고 텍스트만 추출
today_texts = [text.strip() for text in today_data if text.strip()]

tfidf_vectorizer.fit(today_texts)
vector = tfidf_vectorizer.transform(today_texts).toarray()
vector = np.array(vector)

# 2. DBSCAN Clustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

model = DBSCAN(eps=0.1, min_samples=1, metric='cosine')
# Cosine Similarity를 사용하기 위해 metric='cosine'으로 설정
# eps는 epsilon의 약자로, 반경을 의미, min_samples는 클러스터의 최소 크기를 의미
# min_samples 값이 높을 수록 군집으로 판단하는 기준이 까다로워짐
# eps 값이 클수록 많은 데이터를 하나의 군집으로 판단
result = model.fit_predict(vector)

print('클러스터 개수 : ', result.max())

def extract_representative_articles(cluster_labels, texts, num_articles=None):
    representative_articles = []
    
    unique_labels = np.unique(cluster_labels)

    for label in unique_labels:
        print(f'cluster num : {label+1}') # 클러스터 번호 출력
        cluster_indices = np.where(cluster_labels == label)[0]
        cluster_texts = [texts[i] for i in cluster_indices]
        
        # 대표 기사 추출
        representative_article = max(cluster_texts, key=len)
        print(representative_article)
        
        # 해당 클러스터의 기타 기사 출력
        print('기타 기사:')
        for other_article in cluster_texts:
            if other_article != representative_article:
                print(other_article)
        
        # 대표 기사 리스트에 추가
        representative_articles.append(representative_article)
        
        if num_articles is not None and len(representative_articles) >= num_articles:
            break

    return representative_articles

key_model = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2')
def keyword(data, col_cluster):  #data = cluster_result (데이터프레임) #1분 30초 소요됨
    result = []
    for i in range(len(data)):
        key_text = cluster1_result[cluster1_result[col_cluster]==i]['noun']
        key_text = ' '.join(key_text)
        keyword = key_model.extract_keywords(key_text, keyphrase_ngram_range=(1,2), top_n=1)
        result.append(keyword[0][0])
    return result

def merge_keyword(data, col_cluster): #새 열로 추가.
    data_temp = data.copy()
    data_temp['keyword'] = keyword(data, col_cluster)
    return data_temp

keyword_result = merge_keyword(cluster2_result, col_cluster='cluster2nd')

keyword_df = keyword_result[['cluster_num', 'count', 'keyword']]
keyword_df.sort_values(by='count', ascending=False, inplace=True, ignore_index=True)
keyword_df.drop(index=[0], inplace=True)
keyword_df = keyword_df[keyword_df['count']>5]
lst = []
for i in keyword_df['keyword']:
    lst.append(i.upper())
keyword_df['keyword'] = lst

def save_results_to_txt(cluster_labels, texts, filepath, num_articles=None):
    with open(filepath, 'w', encoding='utf-8') as file:
        unique_labels = np.unique(cluster_labels)

        for label in unique_labels:
            file.write(f'cluster num : {label+1}\n')  # 클러스터 번호 출력
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_texts = [texts[i] for i in cluster_indices]
            
            # 대표 기사 추출
            representative_article = max(cluster_texts, key=len)
            file.write(f"{representative_article}\n")
            
            # 해당 클러스터의 기타 기사 출력
            file.write('기타 기사:\n')
            for other_article in cluster_texts:
                if other_article != representative_article:
                    file.write(f"{other_article}\n")
            
            if num_articles is not None and label+1 >= num_articles:
                break

            file.write("\n")
            
def visualize_clusters(vector, cluster_labels):
    # 차원 축소
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(vector)

    # 데이터프레임 생성
    df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
    df['cluster'] = cluster_labels

    # 시각화
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=df, palette='Set1', alpha=0.7)
    plt.title('K-means Clustering with 2D PCA')
    plt.show()

def determine_cluster_count(vector):
    # 클러스터 개수 범위 설정
    min_clusters = 2
    max_clusters = 200
    
    # 각 클러스터 개수에 대한 inertia 값을 저장할 리스트
    inertias = []
    
    for n_clusters in range(min_clusters, max_clusters+1):
        kmeans_model = KMeans(n_clusters=n_clusters)
        kmeans_model.fit(vector)
        inertias.append(kmeans_model.inertia_)
    
    # inertia 값을 시각화하여 클러스터 개수 결정
    plt.plot(range(min_clusters, max_clusters+1), inertias, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Inertia vs Number of Clusters')
    plt.show()
    
# determine_cluster_count(vector)

kmeans_model = KMeans(n_clusters=200)
kmeans_result = kmeans_model.fit_predict(vector)

save_results_to_txt(kmeans_result, today_texts, "/Users/jack/Documents/GitHub/MakeSense/AI_model/representative_articles.txt", num_articles=kmeans_model.n_clusters)
visualize_clusters(vector, kmeans_result)
print(keyword_df)
# representative_articles = extract_representative_articles(kmeans_result, today_texts, num_articles=kmeans_model.n_clusters)