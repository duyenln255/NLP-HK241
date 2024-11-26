import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import numpy as np
from gensim.models import Word2Vec

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Đọc dữ liệu từ file
quote_df = pd.read_csv("Quote_updated.csv")
features_df = pd.read_csv("2_3_Extracted_Features.csv")

# Ghép hai file dữ liệu lại với nhau (nối theo tên tác giả)
merged_df = pd.merge(
    quote_df, features_df, left_on="Author", right_on="Author", how="inner"
)

# Trích xuất POS tagging từ cột "Quote"
def extract_pos_features(text):
    doc = nlp(text)
    pos_counts = {pos: 0 for pos in ["NOUN", "VERB", "ADJ", "ADV"]}
    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
    return [pos_counts["NOUN"], pos_counts["VERB"], pos_counts["ADJ"], pos_counts["ADV"]]

merged_df[["NOUN", "VERB", "ADJ", "ADV"]] = merged_df["Quote"].apply(
    lambda x: pd.Series(extract_pos_features(str(x)))
)

# Trích xuất Word Embeddings với Word2Vec
print("Training Word2Vec model...")
quotes = merged_df["Quote"].apply(lambda x: str(x).split())
word2vec_model = Word2Vec(sentences=quotes, vector_size=100, window=5, min_count=1, workers=4)

def get_average_word2vec(text):
    words = str(text).split()
    vecs = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

# Tạo cột Word2Vec
merged_df["Word2Vec"] = merged_df["Quote"].apply(get_average_word2vec)

# Chuyển đổi Word2Vec thành DataFrame
word2vec_features = pd.DataFrame(
    merged_df["Word2Vec"].tolist(),  # Chuyển mỗi vector thành một dòng
    index=merged_df.index
)

# Gắn thêm các đặc trưng Word2Vec vào DataFrame
X = pd.concat([merged_df.drop(columns=["Author", "Link", "DateofBirth", "Quote", "Word2Vec"]), word2vec_features], axis=1)

# Chuyển đổi nhãn (Author) thành số
label_encoder = LabelEncoder()
merged_df["AuthorEncoded"] = label_encoder.fit_transform(merged_df["Author"])
y = merged_df["AuthorEncoded"]

# Chuyển đổi tất cả tên cột của X sang kiểu str
X.columns = X.columns.astype(str)

# Chia dữ liệu thành tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. SVM
print("\nTraining SVM...")
svm = SVC(kernel="linear", probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Lấy tất cả các nhãn có trong tập huấn luyện
all_labels = sorted(set(y_train) | set(y_test))

# Classification Report cho SVM
print("\nSVM Classification Report:")
svm_report = classification_report(
    y_test,
    y_pred_svm,
    labels=all_labels,
    target_names=label_encoder.inverse_transform(all_labels),
    zero_division=0
)
print(svm_report)

# Lưu Classification Report vào file CSV
svm_report_dict = classification_report(
    y_test,
    y_pred_svm,
    labels=all_labels,
    target_names=label_encoder.inverse_transform(all_labels),
    output_dict=True,
    zero_division=0
)
svm_report_df = pd.DataFrame(svm_report_dict).transpose()
svm_report_df.to_csv("2_4_1_SVM_Classification_Report.csv", index=True, encoding="utf-8")
print("SVM Classification Report saved to '2_4_1_SVM_Classification_Report.csv'.")

# 2. Hiển thị Confusion Matrix của SVM
conf_matrix_svm = confusion_matrix(
    y_test,
    y_pred_svm,
    labels=all_labels
)

plt.figure(figsize=(12, 10))
sns.heatmap(
    conf_matrix_svm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.inverse_transform(all_labels),
    yticklabels=label_encoder.inverse_transform(all_labels),
    cbar=False
)
plt.title("Confusion Matrix (SVM)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("2_4_1_SVM_Confusion_Matrix.png")
plt.show()

print("SVM Confusion Matrix saved as '2_4_1_SVM_Confusion_Matrix.png'.")
