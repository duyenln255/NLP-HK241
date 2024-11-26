import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Đọc dữ liệu từ file Quote_updated.csv
df = pd.read_csv("Quote_updated.csv")

# Loại bỏ khoảng trắng dư thừa trong cột "Author"
df["Author"] = df["Author"].str.strip()

# Xử lý các giá trị thiếu trong cột "Quote"
df["Quote"] = df["Quote"].fillna("")

# 1. TF-IDF Features
tfidf = TfidfVectorizer(stop_words="english", max_features=50)  # Giới hạn 50 từ phổ biến nhất
tfidf_matrix = tfidf.fit_transform(df["Quote"])

# Chuyển đổi ma trận TF-IDF thành DataFrame
tfidf_features = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out(), index=df["Author"])

# Tổng hợp TF-IDF theo từng tác giả
tfidf_by_author = tfidf_features.groupby("Author").sum()

# 2. Statistical Features
df["QuoteLength"] = df["Quote"].apply(len)  # Độ dài câu nói (số ký tự)
df["WordCount"] = df["Quote"].apply(lambda x: len(str(x).split()))  # Số từ trong câu nói

# Tổng hợp đặc trưng thống kê theo từng tác giả
statistical_features = df.groupby("Author").agg({
    "QuoteLength": ["mean", "std", "max"],
    "WordCount": ["mean", "std", "max"]
}).reset_index()

# Đặt lại tên cột
statistical_features.columns = ["Author", "AvgQuoteLength", "StdQuoteLength", "MaxQuoteLength",
                                "AvgWordCount", "StdWordCount", "MaxWordCount"]

# Xử lý các giá trị thiếu trong dữ liệu thống kê
statistical_features = statistical_features.fillna(0)  # Thay thế giá trị `NaN` bằng 0

# 3. Temporal Features
df["YearOfBirth"] = pd.to_datetime(df["DateofBirth"], errors="coerce").dt.year
temporal_features = df.groupby("Author")["YearOfBirth"].mean().reset_index()
temporal_features.columns = ["Author", "AvgYearOfBirth"]

# Xử lý các giá trị thiếu trong cột "YearOfBirth"
temporal_features["AvgYearOfBirth"] = temporal_features["AvgYearOfBirth"].fillna(0)

# 4. Kết hợp tất cả các đặc trưng
# Kết hợp thống kê và thời gian trước
combined_features = statistical_features.merge(temporal_features, on="Author", how="left")

# Kết hợp với TF-IDF
final_features = combined_features.merge(tfidf_by_author, on="Author", how="left")

# Kiểm tra và thay thế các giá trị `NaN` trong TF-IDF
final_features = final_features.fillna(0)

# Lưu kết quả vào file CSV
final_features.to_csv("2_3_Extracted_Features.csv", index=False, encoding="utf-8")
print("Feature extraction completed and saved to '2_3_Extracted_Features.csv'.")
