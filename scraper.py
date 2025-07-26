# scraper.py
import pyodbc
from google_play_scraper import reviews, Sort
import time

# --- 1. Database Connection Configuration ---
# Update these details to match your SQL Server setup
SERVER = 'localhost'  # Replace with your server name/IP if not local
DATABASE = 'PlayStoreReviewsDB'
# Use Windows Authentication or SQL Server Authentication
# For Windows Authentication (if running on the same machine/domain):
CONNECTION_STRING = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;'
# For SQL Server Authentication (replace username/password):
# USERNAME = 'your_username'
# PASSWORD = 'your_password'
# CONNECTION_STRING = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD}'

# --- 2. Define Target Apps ---
APPS = {
    "AliExpress": "com.alibaba.aliexpresshd",
    "Alibaba": "com.alibaba.intl.android.apps.poseidon", # Verify this ID
    "Jiji": "ng.jiji.app" # Verify this ID
}

# --- 3. Database Connection Function ---
def get_db_connection():
    """Establishes and returns a connection to the SQL Server database."""
    try:
        conn = pyodbc.connect(CONNECTION_STRING)
        print("Successfully connected to SQL Server database")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None # Return None to indicate failure

# --- 4. Scraping and Insertion Function ---
def scrape_and_store_reviews(app_name, app_id, conn, max_reviews=1000):
    """
    Scrapes reviews using google-play-scraper and inserts them into the database.
    """
    print(f"\n--- Starting to scrape reviews for {app_name} ({app_id}) ---")
    cursor = conn.cursor()

    # Initial request
    result, continuation_token = reviews(
        app_id,
        lang='en',
        country='us',
        sort=Sort.NEWEST, # Start with newest
        count=min(100, max_reviews) # Max 100 per request
    )

    total_reviews_scraped = 0
    batch_number = 1

    while result and total_reviews_scraped < max_reviews:
        print(f"  Processing batch {batch_number} for {app_name}...")
        for review in result:
            play_store_review_id = review.get('reviewId')
            user_name = review.get('userName')
            user_image_url = review.get('userImage') # Might be None
            review_text = review.get('content')
            rating = review.get('score')
            thumbs_up_count = review.get('thumbsUpCount')
            app_version = review.get('reviewCreatedVersion')
            review_date = review.get('at') # This is a datetime object
            reply_content = review.get('replyContent')
            replied_at = review.get('repliedAt')

            # SQL INSERT statement
            insert_query = """
            INSERT INTO AppReviews
            (AppName, PlayStoreReviewID, UserName, UserImageURL, ReviewText, Rating,
             ThumbsUpCount, AppVersion, ReviewDate, ReplyContent, RepliedAt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            try:
                # Execute the INSERT
                cursor.execute(insert_query, app_name, play_store_review_id, user_name,
                               user_image_url, review_text, rating, thumbs_up_count,
                               app_version, review_date, reply_content, replied_at)
                # It's often better to commit in batches for performance, but committing each row is safer for small datasets/learning
                conn.commit()
                total_reviews_scraped += 1

            except pyodbc.IntegrityError as e:
                # This usually happens if the UNIQUE constraint on PlayStoreReviewID fails (duplicate)
                if "duplicate" in str(e).lower() or "unique" in str(e).lower():
                    print(f"    Duplicate Review ID {play_store_review_id} found for {app_name}, skipping.")
                else:
                    print(f"    Integrity Error inserting review {play_store_review_id}: {e}")
            except Exception as e:
                print(f"    General Error inserting review {play_store_review_id}: {e}")
                # Depending on the error, you might want to stop or continue

        print(f"    Inserted batch {batch_number}. Reviews so far: {total_reviews_scraped}")

        # Check if we have reached the limit or there are no more reviews
        if continuation_token is None or total_reviews_scraped >= max_reviews:
            break

        # Pause between requests to be respectful
        print("  Pausing for 1 second...")
        time.sleep(1)

        # Fetch the next batch
        remaining_reviews = max_reviews - total_reviews_scraped
        result, continuation_token = reviews(
            app_id,
            continuation_token=continuation_token, # Use the token for next page
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=min(100, remaining_reviews)
        )
        batch_number += 1

    print(f"--- Finished scraping for {app_name}. Total reviews scraped: {total_reviews_scraped} ---")

# --- 5. Main Execution Function ---
def main():
    """Main function to orchestrate the scraping process for all apps."""
    print("Starting the Google Play Store Review Scraping Project...")
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to the database. Exiting.")
        return

    try:
        for app_name, app_id in APPS.items():
            # Scrape and store reviews for each app
            # Adjust max_reviews as needed (e.g., 500 or 1000 per app for testing)
            scrape_and_store_reviews(app_name, app_id, conn, max_reviews=500) # Start with 500 for testing

    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during scraping: {e}")
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

# --- 6. Run the Script ---
if __name__ == "__main__":
    main()
    print("\nScraping script finished.")