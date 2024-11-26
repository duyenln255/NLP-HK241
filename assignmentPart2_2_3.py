# import pandas as pd
# from collections import Counter
#
# # Read the dataset
# df = pd.read_csv("Quote_updated.csv")
#
# # 1. Calculate quote lengths
# df["QuoteLength"] = df["Quote"].apply(len)
#
# # Longest and shortest quotes
# longest_quote = df.loc[df["QuoteLength"].idxmax()]
# shortest_quote = df.loc[df["QuoteLength"].idxmin()]
#
# # Prepare longest and shortest quotes for saving
# quote_stats = pd.DataFrame([
#     {"Metric": "Longest Quote", "Quote": longest_quote["Quote"], "Length": longest_quote["QuoteLength"]},
#     {"Metric": "Shortest Quote", "Quote": shortest_quote["Quote"], "Length": shortest_quote["QuoteLength"]}
# ])
#
# # Save to CSV
# quote_stats.to_csv("2_2_3Quote_Statistics.csv", index=False, encoding="utf-8")
#
# # 2. Calculate word count for each quote
# df["WordCount"] = df["Quote"].apply(lambda x: len(str(x).split()))
#
# # Word count statistics
# word_count_summary = df["WordCount"].describe()
#
# # Convert word count statistics to DataFrame
# word_count_summary_df = pd.DataFrame(word_count_summary).reset_index()
# word_count_summary_df.columns = ["Metric", "Value"]
#
# # Save word count statistics to CSV
# word_count_summary_df.to_csv("2_2_3Word_Count_Summary.csv", index=False, encoding="utf-8")
#
# # 3. Analyze word frequencies
# word_list = " ".join(df["Quote"]).split()
# word_counts = Counter(word_list)
# most_common_words = pd.DataFrame(word_counts.most_common(10), columns=["Word", "Frequency"])
#
# # Save most common words to CSV
# most_common_words.to_csv("2_2_3Most_Common_Words.csv", index=False, encoding="utf-8")
#
# print("Statistics successfully saved to CSV files.")

import pandas as pd
from tabulate import tabulate
from collections import Counter

# Read the dataset
df = pd.read_csv("Quote_updated.csv")

# 1. Calculate quote lengths
df["QuoteLength"] = df["Quote"].apply(len)

# Longest and shortest quotes
longest_quote = df.loc[df["QuoteLength"].idxmax()]
shortest_quote = df.loc[df["QuoteLength"].idxmin()]

# 2. Calculate word count for each quote
df["WordCount"] = df["Quote"].apply(lambda x: len(str(x).split()))

# 3. Analyze word frequencies
word_list = " ".join(df["Quote"]).split()
word_counts = Counter(word_list)
most_common_words = word_counts.most_common(10)

# Display results
# Summary of quote statistics
quote_stats = [
    ["Longest Quote", longest_quote["Quote"], longest_quote["QuoteLength"]],
    ["Shortest Quote", shortest_quote["Quote"], shortest_quote["QuoteLength"]]
]

print("\nStatistics about the Quotes:")
print(tabulate(quote_stats, headers=["Metric", "Quote", "Length"], tablefmt="grid"))

# Summary of word count distribution
word_count_stats = df["WordCount"].describe()
word_count_summary = word_count_stats.reset_index().values.tolist()

print("\nWord Count Distribution:")
print(tabulate(word_count_summary, headers=["Metric", "Value"], tablefmt="grid"))

# Most common words
print("\nMost Common Words in Quotes:")
common_words_table = pd.DataFrame(most_common_words, columns=["Word", "Frequency"])
print(tabulate(common_words_table, headers="keys", tablefmt="grid"))
