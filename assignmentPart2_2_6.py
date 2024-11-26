import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("Quote_updated.csv")

# Chuẩn bị dữ liệu: Gom nhóm các câu nói theo tác giả
author_quotes = df.groupby("Author")["Quote"].apply(lambda x: " ".join(x)).reset_index()

# Tính TF-IDF cho các câu nói của từng tác giả
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(author_quotes["Quote"])

# Tính cosine similarity giữa các tác giả
cosine_sim = cosine_similarity(tfidf_matrix)

# Tạo DataFrame từ ma trận similarity
cosine_df = pd.DataFrame(cosine_sim, index=author_quotes["Author"], columns=author_quotes["Author"])

# Lưu ma trận similarity vào file CSV
cosine_df.to_csv("2_2_6_Author_Similarity_Matrix.csv")
print("Cosine similarity matrix saved as '2_2_6_Author_Similarity_Matrix.csv'.")

# Tạo biểu đồ mạng (Graph)
# Xây dựng đồ thị với networkx
graph = nx.Graph()

# Thêm các node (tác giả) và các cạnh (mức độ tương đồng > 0.2)
threshold = 0.2  # Ngưỡng để tạo kết nối
for i, author1 in enumerate(author_quotes["Author"]):
    for j, author2 in enumerate(author_quotes["Author"]):
        if i != j and cosine_sim[i, j] > threshold:  # Không kết nối tác giả với chính mình
            graph.add_edge(author1, author2, weight=cosine_sim[i, j])

# Vẽ biểu đồ mạng
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(graph, seed=42)  # Định vị các node
edges = graph.edges(data=True)

# Vẽ các cạnh với độ rộng tỉ lệ theo mức độ tương đồng
nx.draw_networkx_edges(
    graph, pos, edgelist=edges, width=[d["weight"] * 5 for (_, _, d) in edges], alpha=0.6
)

# Vẽ các node
nx.draw_networkx_nodes(graph, pos, node_size=700, node_color="skyblue", alpha=0.9)

# Vẽ nhãn của các node
nx.draw_networkx_labels(graph, pos, font_size=10, font_color="black")

plt.title("Author Relationship Based on Quote Similarity")
plt.tight_layout()
plt.savefig("2_2_6_Author_Relationship_Graph.png")  # Lưu biểu đồ
plt.show()
