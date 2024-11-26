import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from itertools import combinations

# Đọc dữ liệu
quote_df = pd.read_csv("Quote_updated.csv")
features_df = pd.read_csv("2_3_Extracted_Features.csv")

# Kết hợp dữ liệu
merged_df = pd.merge(quote_df, features_df, on="Author", how="inner")

# Lấy danh sách các tác giả
authors = merged_df["Author"].unique()

# 1. Tính toán similarity dựa trên nội dung câu nói (TF-IDF)
print("Calculating TF-IDF similarity...")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(merged_df["Quote"])  # Vector hóa các câu nói

# Tính toán similarity giữa các tác giả dựa trên TF-IDF
author_quotes = {author: " ".join(merged_df[merged_df["Author"] == author]["Quote"])
                 for author in authors}
author_tfidf_matrix = tfidf.transform(list(author_quotes.values()))
tfidf_sim_matrix = cosine_similarity(author_tfidf_matrix)

# 2. Tính toán similarity dựa trên các đặc trưng số lượng
print("Calculating numerical feature similarity...")
numerical_features = features_df.drop(columns=["Author"])  # Loại bỏ cột "Author"
author_features = numerical_features.groupby(features_df["Author"]).mean()  # Trung bình các đặc trưng theo tác giả

# Tính toán similarity dựa trên Euclidean distance
distance_matrix = np.zeros((len(authors), len(authors)))
for i, j in combinations(range(len(authors)), 2):
    dist = euclidean(author_features.iloc[i], author_features.iloc[j])
    distance_matrix[i, j] = dist
    distance_matrix[j, i] = dist

# Chuyển Euclidean distance thành similarity (1 / (1 + distance))
numerical_sim_matrix = 1 / (1 + distance_matrix)

# 3. Kết hợp similarity từ TF-IDF và đặc trưng số
print("Combining similarities...")
combined_similarity = (tfidf_sim_matrix + numerical_sim_matrix) / 2

# 4. Tìm cặp tác giả có phong cách tương đồng nhất
print("Finding most similar authors...")
max_similarity = -1
most_similar_pair = None
for i in range(len(authors)):
    for j in range(i + 1, len(authors)):
        if combined_similarity[i, j] > max_similarity:
            max_similarity = combined_similarity[i, j]
            most_similar_pair = (authors[i], authors[j])

# Xuất kết quả
print(f"Most similar authors: {most_similar_pair} with similarity score: {max_similarity:.4f}")
similarity_df = pd.DataFrame(
    combined_similarity, index=authors, columns=authors
)
similarity_df.to_csv("2_4_2_Author_Similarity_Matrix.csv", encoding="utf-8")
print("Author similarity matrix saved to '2_4_2_Author_Similarity_Matrix.csv'.")
