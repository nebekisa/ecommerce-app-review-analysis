# Project Methodology: Google Play Store E-commerce App Review Analysis

This document outlines the methodology, tools, and techniques employed in each stage of the "Google Play Store E-commerce App Review Analysis" project. It details the approach taken for data acquisition, storage, cleaning, analysis, modeling, and dashboard development.

## 1. Data Acquisition & Storage

### 1.1. Data Source
Reviews were scraped from the Google Play Store for three popular e-commerce applications: AliExpress, Alibaba, and Jiji. The focus was on collecting English reviews, including the review text, star rating, review date, and associated metadata.

### 1.2. Scraping Methodology
*   **Library Choice:** The `google-play-scraper` Python library was initially evaluated and found to be functional. This library provides a direct and efficient way to access Play Store review data without needing to handle complex browser automation for this specific task.
*   **Implementation:** A Python script (`scraper.py`) was developed using `google-play-scraper`. It was configured to fetch reviews sorted by 'Newest' to capture recent feedback.
*   **Pagination Handling:** The library's built-in support for continuation tokens was utilized to iterate through multiple pages of reviews, ensuring a substantial number of reviews (targeting 500 per app) were collected.
*   **Rate Limiting:** A `time.sleep(1)` delay was implemented between batch requests to be respectful of the Play Store's servers and avoid potential rate limiting.
*   **Challenges & Considerations:**
    *   *Backup Plan:* Selenium web scraping was researched and prepared as a backup method (involving ChromeDriver setup) in case `google-play-scraper` failed, but it was not required for the final execution.
    *   *Data Completeness:* The scraper aimed to collect core review elements (text, rating, date) along with app-specific and user-specific identifiers where available.

### 1.3. Database Storage
*   **Database System:** Microsoft SQL Server was chosen for data storage, managed using SQL Server Management Studio (SSMS).
*   **Schema Design:** A relational database schema (`PlayStoreReviewsDB`) was designed with a primary table `dbo.AppReviews`.
*   **Table Structure:** The `AppReviews` table included columns for:
    *   `ReviewID` (Primary Key, auto-incrementing integer)
    *   `AppName` (NVARCHAR)
    *   `PlayStoreReviewID` (NVARCHAR, Unique)
    *   `UserName` (NVARCHAR)
    *   `UserImageURL` (NVARCHAR)
    *   `ReviewText` (NVARCHAR(MAX))
    *   `Rating` (INT, with CHECK constraint 1-5)
    *   `ThumbsUpCount` (INT)
    *   `AppVersion` (NVARCHAR)
    *   `ReviewDate` (DATETIME2)
    *   `ReplyContent` (NVARCHAR(MAX))
    *   `RepliedAt` (DATETIME2)
    *   `ScrapedAt` (DATETIME2, defaulting to current timestamp)
*   **Connection:** The Python script used `pyodbc` to connect to the SQL Server instance and insert the scraped review data into the `AppReviews` table. Unique constraints helped prevent duplicate entries.

## 2. Data Cleaning & Preprocessing

### 2.1. Data Loading
*   The raw review data was loaded from the `dbo.AppReviews` table in SQL Server into a Pandas DataFrame using `pyodbc` within a Jupyter Notebook (`01_data_cleaning_preprocessing.ipynb`).

### 2.2. Data Cleaning Steps
*   **Missing Text Handling:** Reviews with missing, empty, or whitespace-only `ReviewText` were identified and removed from the dataset as they were not useful for text analysis.
*   **Data Type Verification:** Ensured `ReviewDate` was in datetime format.

### 2.3. Text Preprocessing (NLP)
A dedicated function (`clean_and_preprocess_text`) was implemented to standardize the review text:
*   **Lowercasing:** All text was converted to lowercase.
*   **HTML Tag Removal:** Used `BeautifulSoup` to strip HTML tags.
*   **URL Removal:** Applied regular expressions (`re`) to remove web addresses.
*   **Emoji Removal:** Used a regular expression pattern to identify and remove emojis.
*   **Contraction Expansion:** Utilized the `contractions` library to expand common contractions (e.g., "don't" to "do not").
*   **Punctuation/Number Removal:** Removed punctuation and numerical digits, keeping only alphabetic characters and spaces.
*   **Tokenization:** Split text into individual words using `nltk.tokenize.word_tokenize`.
*   **Stopword Removal:** Removed common English stopwords using `nltk.corpus.stopwords`.
*   **Lemmatization:** Reduced words to their base/dictionary form using `nltk.stem.WordNetLemmatizer`.
*   **Result Storage:** The cleaned text was stored in a new column `CleanedReviewText`.

### 2.4. Feature Engineering
*   **Review Length:** Calculated the word count of the cleaned review text and stored it in a new column `ReviewWordCount`.

### 2.5. Data Saving
*   The final cleaned DataFrame, including original columns, `CleanedReviewText`, and `ReviewWordCount`, was saved to a CSV file (`cleaned_reviews_data.csv`) for use in subsequent stages.

## 3. Exploratory Data Analysis (EDA)

### 3.1. Environment
*   Analysis was conducted using Jupyter Notebooks (`02_eda_insights.ipynb`).

### 3.2. Libraries Used
*   `pandas`, `numpy` for data manipulation.
*   `matplotlib`, `seaborn`, `plotly` for static and interactive visualizations.
*   `wordcloud` for generating word clouds.
*   `vaderSentiment` for baseline sentiment analysis.
*   `scikit-learn` (`CountVectorizer`) for N-gram analysis.
*   `nltk` for text processing utilities.

### 3.3. EDA Activities Performed
*   **Review Count Analysis:** Analyzed total reviews per app and visualized trends over time (monthly/yearly) using line plots.
*   **Rating Distribution Analysis:** Examined the distribution of 1-5 star ratings per app using count plots and bar charts for average ratings.
*   **Review Length Analysis:** Investigated the distribution of review word counts using histograms and box plots, and explored relationships with app and rating.
*   **Common Words/N-grams Analysis:** Identified and visualized the most frequent unigrams, bigrams, and trigrams for each app using bar charts and word clouds.
*   **Basic Sentiment Overview:** Applied VADER sentiment analysis to `CleanedReviewText` to classify reviews as Positive, Negative, or Neutral. Visualized sentiment distribution per app and trends over time. Correlated VADER scores with star ratings.
*   **Missing Value Analysis:** Checked for and visualized any remaining missing data points in the cleaned dataset.

## 4. Statistical Modeling / Machine Learning

### 4.1. Objective
*   To build and evaluate machine learning models for classifying the sentiment (Positive, Negative, Neutral) of a review based solely on its text (`CleanedReviewText`).

### 4.2. Environment & Libraries
*   Jupyter Notebook (`03_modeling_sentiment.ipynb`).
*   `scikit-learn` for modeling, vectorization, evaluation, and pipelines.
*   `joblib` for saving the trained model.

### 4.3. Data Preparation
*   Loaded the `cleaned_reviews_data.csv`.
*   Defined features (`X`) as `CleanedReviewText` and target (`y`) as `VADER_Sentiment`.
*   Split the data into training (80%) and testing (20%) sets using `train_test_split` with `stratify=y` to maintain class distribution.

### 4.4. Feature Engineering for Modeling
*   Utilized `sklearn.feature_extraction.text.TfidfVectorizer` within pipelines to convert text (`X_train`, `X_test`) into numerical TF-IDF features.
    *   Parameters: `max_features=5000`, `ngram_range=(1, 2)`, `stop_words='english'`.

### 4.5. Model Selection & Training
*   Evaluated three classification algorithms using `sklearn.pipeline.Pipeline`:
    *   `LogisticRegression`
    *   `RandomForestClassifier`
    *   `MultinomialNB`
*   Each pipeline combined the `TfidfVectorizer` and the respective classifier.
*   Models were trained on the `X_train`, `y_train` set.

### 4.6. Model Evaluation
*   Models were evaluated on the unseen `X_test` set.
*   Predictions (`y_pred`) were compared to true labels (`y_test`).
*   Primary metric: Accuracy (`sklearn.metrics.accuracy_score`).
*   Detailed analysis for the best model (based on accuracy):
    *   Classification Report (`sklearn.metrics.classification_report`): Precision, Recall, F1-Score per class.
    *   Confusion Matrix (`sklearn.metrics.confusion_matrix`): Visualized using `seaborn.heatmap`.
    *   Feature Importance: Analyzed coefficients from the best `LogisticRegression` model to understand key words/phrases for each sentiment class.
*   **Class Imbalance Handling:** Recognized the imbalance (Positive >> Negative >> Neutral). Added a variant `Logistic Regression Balanced` using `class_weight='balanced'` to improve performance for minority classes.

### 4.7. Model Persistence
*   The best-performing model pipeline was saved using `joblib.dump` for potential future use.

## 5. Dashboard Development

### 5.1. Technology
*   Streamlit was chosen for its ease of creating interactive web applications directly from Python scripts.

### 5.2. Implementation
*   Developed `dashboard.py` as the main Streamlit application script.
*   Loaded `cleaned_reviews_data.csv` for visualization.
*   Implemented interactive elements:
    *   `st.sidebar.selectbox` for app selection.
*   Created visualizations using `plotly` and `matplotlib`/`wordcloud`:
    *   Overview statistics using `st.metric`.
    *   Interactive charts (bar, pie, line) for distributions and trends.
    *   Tabs for different word analysis views (Unigrams, Bigrams, Word Cloud).
    *   Expanders for displaying sample reviews.
*   Structured the layout using `st.columns` and `st.tabs` for a professional appearance.
*   Applied custom CSS for minor styling improvements.

### 5.3. Features Implemented (Meeting Minimum Requirements)
*   **App Selection:** Dropdown to choose between AliExpress, Alibaba, Jiji.
*   **Overview Statistics:** Displays total reviews, average rating, positive sentiment %, dominant sentiment for the selected app.
*   **Rating Distribution Visualization:** Interactive bar chart showing 1-5 star breakdown.
*   **Sentiment Distribution Visualization:** Pie chart showing Positive/Neutral/Negative breakdown.
*   **Key Topics/Words Visualization:** Tabs for Top Unigrams, Top Bigrams (bar charts), and Word Cloud.
*   **Sentiment Trend:** Line chart showing sentiment distribution over time (monthly) for the selected app.
*   **Example Reviews:** Tabs for Positive/Negative/Neutral, showing sample reviews (with expanders) for the selected app.
