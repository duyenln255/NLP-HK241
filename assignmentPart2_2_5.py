from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

# Đọc dữ liệu
df = pd.read_csv("Quote_updated.csv")

# Tính TF-IDF cho các câu nói
tfidf = TfidfVectorizer(stop_words="english", max_features=50)  # Giới hạn 50 từ
tfidf_matrix = tfidf.fit_transform(df["Quote"])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# Tính tổng TF-IDF score cho từng từ theo tác giả
df["Author"] = df["Author"].str.strip()  # Xử lý khoảng trắng dư thừa
authors_tfidf = df.groupby("Author").apply(
    lambda x: tfidf_df.loc[x.index].sum(axis=0)
).transpose()

# Chọn một số tác giả tiêu biểu để so sánh
selected_authors = authors_tfidf.columns[:5]  # Top 5 tác giả đầu tiên
authors_tfidf[selected_authors].plot(kind="bar", figsize=(12, 6), stacked=True)
plt.title("Key Words Comparison Across Authors")
plt.xlabel("Words")
plt.ylabel("TF-IDF Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("2_2_5_Authors_Keywords_Comparison.png")  # Lưu biểu đồ
plt.show()

# Áp dụng K-Means clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df["Cluster"] = kmeans.fit_predict(tfidf_matrix)

# Giảm chiều dữ liệu để trực quan hóa (sử dụng PCA)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(tfidf_matrix.toarray())
# Vẽ biểu đồ phân cụm
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=reduced_data[:, 0],
    y=reduced_data[:, 1],
    hue=df["Cluster"],
    palette="viridis",
    style=df["Author"],
    legend="full"
)

# Thêm tiêu đề và nhãn trục
plt.title("Authors Clustering Based on Quotes", fontsize=14)
plt.xlabel("PCA Component 1", fontsize=12)
plt.ylabel("PCA Component 2", fontsize=12)

# Điều chỉnh cỡ chữ của chú thích (legend)
legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Legend", fontsize=5, title_fontsize=5)
for text in legend.get_texts():
    text.set_fontsize(5)  # Set font size for legend text
legend.get_title().set_fontsize(5)  # Set font size for legend title

# Lưu biểu đồ
plt.tight_layout()
plt.savefig("2_2_5_Authors_Clustering_Updated.png")
plt.show()
