# test_scraper.py
from google_play_scraper import reviews, Sort

# --- Try scraping reviews for AliExpress ---
# Find the correct app ID (this one is usually right for AliExpress)
app_id = 'com.alibaba.aliexpresshd'

try:
    print(f"Attempting to scrape reviews for app ID: {app_id}")
    # Get the first 10 reviews, sorted by Newest
    result, continuation_token = reviews(
        app_id,
        lang='en', # English reviews
        country='us', # From US Play Store
        sort=Sort.NEWEST, # Sort by newest
        count=10 # Get 10 reviews
    )

    # Print the number of reviews we got
    print(f"Successfully fetched {len(result)} reviews.")
    # Print the first review to see its structure
    if result:
        print("\n--- First Review Data ---")
        print(result[0])
    else:
        print("No reviews were returned.")

except Exception as e:
    print(f"An error occurred while scraping: {e}")
    print("This means the easy method might not work right now.")

print("\nTest finished.")