import pandas as pd
from collections import Counter
import re
from nltk.corpus import stopwords
import nltk
import ssl
from tabulate import tabulate

# Setup for nltk stopwords
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("Quote_updated.csv")

# 1. Tokenize all words from the quotes
word_list = " ".join(df["Quote"]).lower()  # Convert all text to lowercase
word_list = re.findall(r'\b\w+\b', word_list)  # Extract words using regex

# 2. Calculate word frequencies (all words)
word_counts = Counter(word_list)
word_frequency_df = pd.DataFrame(word_counts.most_common(), columns=["Word", "Frequency"])

# Save all word frequencies to a CSV file
word_frequency_df.to_csv("2_2_4_Word_Frequencies.csv", index=False, encoding="utf-8")

# 3. Remove stop words to get meaningful words
stop_words = set(stopwords.words("english"))
filtered_words = [word for word in word_list if word not in stop_words]
filtered_word_counts = Counter(filtered_words)
filtered_word_frequency_df = pd.DataFrame(filtered_word_counts.most_common(), columns=["Word", "Frequency"])

# Save filtered word frequencies to a CSV file
filtered_word_frequency_df.to_csv("2_2_4_Filtered_Word_Frequencies.csv", index=False, encoding="utf-8")

# 4. Calculate average word length
word_lengths = [len(word) for word in word_list]
average_word_length = sum(word_lengths) / len(word_lengths)

# Save word length statistics
word_length_stats = pd.DataFrame(
    [{"Metric": "Total Words", "Value": len(word_list)},
     {"Metric": "Unique Words", "Value": len(word_counts)},
     {"Metric": "Average Word Length", "Value": average_word_length}]
)
word_length_stats.to_csv("2_2_4_Word_Length_Statistics.csv", index=False, encoding="utf-8")

# 5. Print key results in tabular format
print("\nTop 10 Words (with stop words):")
print(tabulate(word_frequency_df.head(10), headers="keys", tablefmt="grid"))

print("\nTop 10 Words (without stop words):")
print(tabulate(filtered_word_frequency_df.head(10), headers="keys", tablefmt="grid"))

print("\nWord Length Statistics:")
print(tabulate(word_length_stats, headers="keys", tablefmt="grid"))
