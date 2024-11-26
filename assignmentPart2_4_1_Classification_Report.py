import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file
quote_df = pd.read_csv("Quote_updated.csv")
features_df = pd.read_csv("2_3_Extracted_Features.csv")

# Ghép hai file dữ liệu lại với nhau (nối theo tên tác giả)
merged_df = pd.merge(
    quote_df, features_df, left_on="Author", right_on="Author", how="inner"
)

# Loại bỏ cột không cần thiết và xử lý dữ liệu
X = merged_df.drop(columns=["Author", "Quote", "Link", "DateofBirth"])  # Chọn đặc trưng
y = merged_df["Author"]  # Nhãn là tên tác giả

# Chuyển đổi nhãn từ chuỗi sang số
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Chia dữ liệu thành tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Data prepared successfully.")
# Xây dựng mô hình Random Forest
clf = RandomForestClassifier(random_state=42, n_estimators=100)
clf.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = clf.predict(X_test)

# Chuyển đổi nhãn số thành tên tác giả
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

print("Model trained and predictions made.")
# Lưu Classification Report vào file CSV
classification_report_dict = classification_report(
    y_test_labels, y_pred_labels, output_dict=True
)
classification_report_df = pd.DataFrame(classification_report_dict).transpose()
classification_report_df.to_csv("2_4_1_Classification_Report.csv", index=True, encoding="utf-8")
print("Classification report saved to '2_4_1_Classification_Report.csv'.")

# Hiển thị Confusion Matrix dưới dạng biểu đồ
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)

plt.figure(figsize=(12, 10))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    cbar=False
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("2_4_1_Confusion_Matrix.png")
plt.show()

print("Confusion matrix saved as '2_4_1_Confusion_Matrix.png'.")

