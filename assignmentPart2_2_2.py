import pandas as pd
from tabulate import tabulate

# Read the dataset
df = pd.read_csv("Quote_updated.csv")

# Ensure DateofBirth is in the correct format (YYYY-MM-DD)
df["DateofBirth"] = pd.to_datetime(df["DateofBirth"], errors="coerce")

# Extract year of birth
df["YearOfBirth"] = df["DateofBirth"].dt.year

# Filter out rows where YearOfBirth is NaN
df_filtered = df.dropna(subset=["YearOfBirth", "Age"])

# Convert Age to numeric (ignoring "Unknown")
df_filtered["Age"] = pd.to_numeric(df_filtered["Age"], errors="coerce")

# 1. Year of Birth Statistics
min_year = int(df_filtered["YearOfBirth"].min())
max_year = int(df_filtered["YearOfBirth"].max())
average_year = int(df_filtered["YearOfBirth"].mean())

# 2. Age Statistics
min_age = int(df_filtered["Age"].min())
max_age = int(df_filtered["Age"].max())
average_age = int(df_filtered["Age"].mean())

# Summary statistics table
summary_table = [
    ["Earliest Year of Birth", min_year],
    ["Latest Year of Birth", max_year],
    ["Average Year of Birth", average_year],
    ["Youngest Age", min_age],
    ["Oldest Age", max_age],
    ["Average Age", average_age]
]

# 3. Distribution of authors by Year of Birth
year_distribution = df_filtered.groupby("YearOfBirth").size().reset_index()
year_distribution.columns = ["YearOfBirth", "Number of Authors"]

# Print the summary statistics
print("Summary Statistics for Year of Birth and Age:")
print(tabulate(summary_table, headers=["Metric", "Value"], tablefmt="grid"))

# Print the distribution of authors by year of birth
print("\nDistribution of Authors by Year of Birth:")
print(tabulate(year_distribution, headers="keys", tablefmt="grid"))
