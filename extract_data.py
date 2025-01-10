import requests
import json
import os
import time
import datetime

# Constants
API_KEY = "your_api_key"
BASE_URL = "https://api.goperigon.com/v1/all?category=Politics&category=Tech&category=Sports&category=Business&category=Finance&category=Entertainment&category=Health&category=Weather&category=Lifestyle&category=Auto&category=Science&category=Travel&category=Environment&category=World&category=General"
OUTPUT_FOLDER = "articles_by_year"

# Function to fetch articles within a date range
def fetch_articles(start_date, end_date):
    params = {
        "apiKey": API_KEY,
        "from": start_date,
        "to": end_date,
    }

    time.sleep(1)  # Pause for 1 second to avoid rate limiting
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return []

# Function to fetch articles for a full year in 5-day intervals
def fetch_articles_for_year(year):
    all_articles = []
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year, 12, 31)

    current_date = start_date
    while current_date <= end_date:
        next_date = current_date + datetime.timedelta(days=4)
        print(f"Fetching articles from {current_date.isoformat()} to {next_date.isoformat()}...")
        articles = fetch_articles(current_date.isoformat(), next_date.isoformat())
        
        # Extract required fields
        filtered_articles = [
            {
                "categories": article.get("categories", []),
                "topics": article.get("topics", []),
                "keywords": article.get("keywords", []),
                "sentiment": article.get("sentiment", {})
            }
            for article in articles
        ]
        all_articles.extend(filtered_articles)

        current_date = next_date + datetime.timedelta(days=1)

    return all_articles

# Function to save articles to a single JSON file
def save_articles_to_file(articles, year):
    # Ensure the output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Define file path
    file_path = os.path.join(OUTPUT_FOLDER, f"articles_{year}.json")
    
    # Save articles to file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(articles)} articles for {year} to {file_path}")

# Main function
def main():
    # Manually select the year
    year = 2018
    print(f"Fetching articles for the year {year}...")
    all_articles = fetch_articles_for_year(year)
    save_articles_to_file(all_articles, year)
    print(f"Finished fetching and saving articles for {year}.")

if __name__ == "__main__":
    main()
