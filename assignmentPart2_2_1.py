import pandas as pd
from tabulate import tabulate

# Read the dataset
df = pd.read_csv("Quote_updated.csv")

# 1. Number of unique authors
unique_authors = df["Author"].nunique()

# 2. Total number of famous quotes
total_quotes = len(df)

# 3. Number of quotes by each author
author_quote_counts = df.groupby("Author").size().sort_values(ascending=False)

# Format the output into tables for better readability
summary_table = [
    ["Unique Authors", unique_authors],
    ["Total Famous Quotes", total_quotes]
]

# Print the summary
print("Summary Statistics:")
print(tabulate(summary_table, headers=["Metric", "Value"], tablefmt="grid"))

# Print the author-wise quote counts
print("\nNumber of Quotes by Each Author:")
author_table = pd.DataFrame(author_quote_counts).reset_index()
author_table.columns = ["Author", "Number of Quotes"]
print(tabulate(author_table, headers="keys", tablefmt="grid"))
