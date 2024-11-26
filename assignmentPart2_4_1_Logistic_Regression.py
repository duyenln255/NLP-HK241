import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Đọc dữ liệu từ file
quote_df = pd.read_csv("Quote_updated.csv")
features_df = pd.read_csv("2_3_Extracted_Features.csv")

# Ghép hai file dữ liệu lại với nhau (nối theo tên tác giả)
merged_df = pd.merge(
    quote_df, features_df, left_on="Author", right_on="Author", how="inner"
)

# Chuyển đổi nhãn (Author) thành số
label_encoder = LabelEncoder()
merged_df["AuthorEncoded"] = label_encoder.fit_transform(merged_df["Author"])

# Tách dữ liệu thành đặc trưng (X) và nhãn (y)
X = merged_df.drop(columns=["Author", "AuthorEncoded", "Link", "DateofBirth"])
y = merged_df["AuthorEncoded"]

# Chia dữ liệu thành tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Trích xuất đặc trưng từ cột "Quote" bằng TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english", max_features=1000)

# Pipeline xử lý dữ liệu
preprocessor = ColumnTransformer(
    transformers=[
        ("tfidf", tfidf, "Quote"),  # Tfidf trên cột "Quote"
        ("numeric", "passthrough", X.drop(columns=["Quote"]).columns)  # Dữ liệu số
    ]
)

# 2. Logistic Regression
print("\nTraining Logistic Regression...")
logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=500, random_state=42))
    ]
)
logistic_pipeline.fit(X_train, y_train)
y_pred_logistic = logistic_pipeline.predict(X_test)

# Lấy tất cả các nhãn có trong tập huấn luyện
all_labels = sorted(set(y_train) | set(y_test))  # Bao gồm cả nhãn không xuất hiện trong y_test

# Classification Report cho Logistic Regression
print("\nLogistic Regression Classification Report:")
logistic_report = classification_report(
    y_test,
    y_pred_logistic,
    labels=all_labels,
    target_names=label_encoder.inverse_transform(all_labels),
    zero_division=0  # Để tránh lỗi chia cho 0
)
print(logistic_report)

# Lưu Classification Report vào file CSV
logistic_report_dict = classification_report(
    y_test,
    y_pred_logistic,
    labels=all_labels,
    target_names=label_encoder.inverse_transform(all_labels),
    output_dict=True,
    zero_division=0
)
logistic_report_df = pd.DataFrame(logistic_report_dict).transpose()
logistic_report_df.to_csv("2_4_1_Logistic_Classification_Report.csv", index=True, encoding="utf-8")
print("Logistic Classification Report saved to '2_4_1_Logistic_Classification_Report.csv'.")

# 3. Hiển thị Confusion Matrix của Logistic Regression
conf_matrix_logistic = confusion_matrix(
    y_test,
    y_pred_logistic,
    labels=all_labels
)

plt.figure(figsize=(12, 10))
sns.heatmap(
    conf_matrix_logistic,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.inverse_transform(all_labels),
    yticklabels=label_encoder.inverse_transform(all_labels),
    cbar=False
)
plt.title("Confusion Matrix (Logistic Regression)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("2_4_1_Logistic_Confusion_Matrix.png")
plt.show()

print("Logistic Confusion Matrix saved as '2_4_1_Logistic_Confusion_Matrix.png'.")
