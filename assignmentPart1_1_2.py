from bs4 import BeautifulSoup

# Đọc dữ liệu từ file kq.txt
with open("kq.txt", "r", encoding="utf-8") as file:
    data = file.read()

# Parse dữ liệu với BeautifulSoup
soup = BeautifulSoup(data, 'html.parser')

# Lấy tất cả thẻ <div> với class "quote" và lưu vào biến result
result = soup.find_all('div', class_='quote')

# Hiển thị giá trị của biến result
print(f"Number of quotes: {len(result)}")
for i, item in enumerate(result[:len(result)]):
    print(f"Quote {i + 1}: {item}\n")
# Lấy danh sách các tác giả từ biến result
authors = [quote.find('small', class_='author').text.strip() for quote in result]

# Hiển thị danh sách các tác giả
print(f"Number of authors: {len(authors)}")
for i, author in enumerate(authors[:len(authors)]):
    print(f"Author {i + 1}: {author}")
import requests
from bs4 import BeautifulSoup


def authorLink(result):
    base_url = "http://quotes.toscrape.com"
    authors_info = []

    for quote in result:
        # Tên tác giả
        author = quote.find('small', class_='author').text.strip()
        # Link chi tiết về tác giả
        author_link = base_url + quote.find('a')['href']
        # Câu nói nổi tiếng
        famous_quote = quote.find('span', class_='text').text.strip()

        # Gửi request đến trang chi tiết của tác giả để lấy ngày sinh
        response = requests.get(author_link)
        author_soup = BeautifulSoup(response.text, 'html.parser')
        # Lấy ngày sinh của tác giả
        dob = author_soup.find('span', class_='author-born-date').text.strip()

        # Lưu thông tin
        authors_info.append({
            "author": author,
            "link": author_link,
            "dob": dob,
            "quote": famous_quote
        })

        # Hiển thị thông tin
        print(f"Author: {author}")
        print(f"Link: {author_link}")
        print(f"Date of Birth: {dob}")
        print(f"Quote: {famous_quote}\n")

    return authors_info


# Gọi hàm để thu thập thông tin
authors_info = authorLink(result)

import csv

# Prepare the data for CSV
csv_data = [
    {
        "Author": item["author"],
        "Link": item["link"],
        "DateofBirth": item["dob"],
        "Quote": item["quote"]
    }
    for item in authors_info  # authors_info already has the correct structure
]

# Write to CSV
with open('Quote.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["Author", "Link", "DateofBirth", "Quote"])
    writer.writeheader()  # Write the header
    writer.writerows(csv_data)  # Write the rows

print("Data successfully written to Quote.csv")
