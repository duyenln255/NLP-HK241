import requests
from bs4 import BeautifulSoup

def crawl_all_html(base_url, output_file):
    """
    Crawl all <div class="quote"> elements from all pages and save them to kq.txt
    """
    current_page = base_url
    quotes_collected = 0

    with open(output_file, 'w', encoding='utf-8') as file:
        while current_page:
            response = requests.get(current_page)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract all <div> elements with class "quote"
            quotes = soup.find_all('div', class_='quote')
            # Save all <div> elements to the file
            for quote in quotes:
                file.write(str(quote) + '\n\n')   # Save raw HTML to file
                quotes_collected += 1

            # Check for "Next" button to navigate to the next page
            next_btn = soup.find('li', class_='next')
            if next_btn:
                next_page = next_btn.find('a')['href']
                current_page = base_url + next_page
            else:
                current_page = None

    print(f"Completed! Total quotes collected: {quotes_collected}")

# Call the function to crawl all data
crawl_all_html("http://quotes.toscrape.com", "kq.txt")
