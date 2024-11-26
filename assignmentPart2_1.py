import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime

# Function to fetch author's Date of Birth and Age directly from Wikipedia
def get_author_details(author_name):
    # Special case: William Nicholson
    if author_name == "William Nicholson":
        base_url = "https://en.wikipedia.org/wiki/William_Nicholson_(writer)"
    else:
        base_url = f"https://en.wikipedia.org/wiki/{author_name.replace(' ', '_')}"
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        birth_date = None
        age = None

        # Extract Born information
        born_section = soup.find("th", string="Born")
        if born_section:
            born_data = born_section.find_next("td")
            birth_date_raw = born_data.find("span", class_="bday")
            if birth_date_raw:
                birth_date = birth_date_raw.text  # Already in YYYY-MM-DD format
            else:
                birth_date_text = born_data.text.strip().split("\n")[0]
                try:
                    birth_date = datetime.strptime(birth_date_text, "%d %B %Y").strftime("%Y-%m-%d")
                except ValueError:
                    birth_date = None

            # Extract age if still alive
            if "age" in born_data.text:
                age_match = re.search(r"\(age\s*(\d+)\)", born_data.text)
                if age_match:
                    age = int(age_match.group(1))

        # Extract Died information (if author is deceased)
        died_section = soup.find("th", string="Died")
        if died_section:
            died_data = died_section.find_next("td").text.strip()
            age_at_death_match = re.search(r"aged\s*(\d+)", died_data)
            if age_at_death_match:
                age = int(age_at_death_match.group(1))

        return birth_date, age

    except Exception as e:
        print(f"Error fetching data for {author_name}: {e}")
        return None, None


# Function to normalize existing DateofBirth to YYYY-MM-DD
def normalize_date(date_string):
    try:
        # Handle existing dates in 'Month DD, YYYY' format
        return datetime.strptime(date_string, "%B %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        # If already in correct format or unrecognized format, return as-is
        return date_string


# Read the Quote.csv file
df = pd.read_csv("Quote.csv")

# Add the "Age" column if it does not exist
if "Age" not in df.columns:
    df["Age"] = "Unknown"

# Normalize existing DateofBirth values
df["DateofBirth"] = df["DateofBirth"].apply(lambda x: normalize_date(x) if pd.notna(x) else x)

# Check and update missing DateofBirth and Age fields
for index, row in df.iterrows():
    author = row["Author"]
    if pd.isna(row["DateofBirth"]) or row["Age"] == "Unknown":  # Only fetch if DateofBirth or Age is missing
        print(f"Fetching data for {author}...")
        birth_date, age = get_author_details(author)
        if birth_date:
            df.at[index, "DateofBirth"] = birth_date
        if age is not None:
            df.at[index, "Age"] = age

# Save updated CSV
df.to_csv("Quote_updated.csv", index=False, encoding="utf-8")
print("Data successfully updated and saved to Quote_updated.csv")
