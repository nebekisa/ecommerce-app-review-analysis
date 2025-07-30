# EDA Summary Report: Google Play Store E-commerce App Reviews

## Introduction

This report summarizes the key findings from the Exploratory Data Analysis (EDA) performed on English reviews collected from the Google Play Store for three popular e-commerce applications: AliExpress, Alibaba, and Jiji. The primary goal of this EDA was to understand the volume, characteristics, sentiment, and common topics discussed in user reviews for these platforms, providing a foundation for subsequent modeling and dashboard development.

## Key Findings

### 1. Review Volume and Trends

*   **Balanced Dataset:** Each app (AliExpress, Alibaba, Jiji) contributed 500 reviews to the dataset, ensuring a fair comparison between platforms.
*   **Temporal Concentration:** A significant surge in review activity was observed for all three apps towards the latter part of the collected data period, particularly in mid-to-late 2025. This indicates a potential event or change affecting user engagement across the board during this timeframe.
*   **App-Specific Growth:** Historical data (though limited for AliExpress) suggests varying growth trajectories:
    *   **AliExpress:** Appears to be a relatively new entrant or underwent a significant relaunch/change around 2023-2024, experiencing rapid recent growth.
    *   **Alibaba & Jiji:** Show evidence of more established user bases with consistent review activity over several years prior to the 2025 surge.

### 2. Rating Patterns

*   **Overall Average Ratings:** Jiji exhibited the highest average rating (approx. 3.36), followed by Alibaba (approx. 2.92), with AliExpress having the lowest (approx. 2.62). The lower rating for AliExpress is likely influenced by its recent entry and associated growing pains.
*   **Rating Distribution Shape:**
    *   **AliExpress & Alibaba:** Showed highly polarized distributions, with substantial peaks at both 1-star and 5-star ratings. This suggests users tend to have extreme opinions (very satisfied or very dissatisfied).
    *   **Jiji:** Displayed a more balanced distribution with fewer extreme ratings and a broader spread across 2-4 stars, indicating a wider range of user experiences or less polarized opinions.

### 3. Review Characteristics (Length)

*   **Average Length:** AliExpress users wrote the longest reviews on average, followed by Alibaba, with Jiji users writing the shortest.
*   **Length vs. Sentiment/Rating:** A consistent pattern emerged where reviews expressing extreme dissatisfaction (1-star) tended to be longer, likely due to detailed explanations. Conversely, extremely positive reviews (5-stars) were often shorter, potentially reflecting quick expressions of satisfaction. AliExpress showed a unique pattern where moderate dissatisfaction (4-star) also correlated with longer reviews.

### 4. Common Topics (Words/N-grams)

Analysis of cleaned review text revealed distinct themes for each app:

*   **AliExpress:**
    *   Frequent terms focused on products (`item`, `product`), sellers (`seller`), shipping/logistics (`shipping`, `refund`, `ordered item`), and the app experience itself (`aliexpress`, `app`).
    *   Bigrams highlighted `customer service` and `shipping cost` as key discussion points.
*   **Alibaba:**
    *   Common terms revolved around the app/platform (`app`, `alibaba`), products (`product`, `supplier`), pricing (`price`, `buy`), and shipping (`shipping`).
    *   Bigrams like `customer service`, `shipping cost`, and `great app` were prominent.
*   **Jiji:**
    *   Discussions frequently mentioned the platform itself (`app`, `jiji`, `ad`) and marketplace interactions (`seller`, `customer`, `buyer`).
    *   Positive sentiment phrases like `good app`, `great app`, `easy use` were common bigrams.
*   **Overall:** Core e-commerce themes like `product`, `price`, `shipping`, `app`, and `customer` were prevalent across all platforms.

### 5. Sentiment Insights (VADER)

Using VADER sentiment analysis on the cleaned text provided nuanced insights:

*   **Overall Sentiment Scores:** Jiji had the highest average VADER sentiment score, followed by AliExpress and then Alibaba. This aligns somewhat with average star ratings.
*   **Sentiment Distribution:** Jiji showed a higher proportion of positive sentiment compared to the others.
*   **Sentiment Correlation with Ratings:** VADER sentiment scores correlated moderately well with star ratings (correlation ~0.57), validating the use of star ratings as a proxy for sentiment but also highlighting the added detail VADER provides.
*   **Temporal Sentiment Trends:** The analysis confirmed the significant increase in review volume in late 2025. The sentiment trend visualization showed how the proportion of positive, negative, and neutral reviews fluctuated over time for each app.

### 6. Data Quality

*   **Missing Data:** The primary dataset (`AppReviews` table in SQL Server) contained minimal missing data for core fields like `ReviewText` and `Rating`. Some fields like `ReplyContent`, `RepliedAt`, and `AppVersion` had a higher frequency of missing values, which is typical (not all reviews receive replies or have version info).
*   **Data Consistency:** The scraping and cleaning process aimed to ensure consistent data types and formats, which was verified during EDA.

## Visualizations

Key visualizations created during EDA included:
*   Bar charts for total review counts and average ratings per app.
*   Line charts for monthly review trends.
*   Histograms and box plots for review length distributions.
*   Count plots for rating distributions.
*   Bar charts for top N-grams (unigrams, bigrams).
*   Word clouds for visualizing prominent terms.
*   Stacked bar charts for sentiment distributions.
*   Line charts for monthly sentiment trends.
*   Box plots showing VADER score distributions across star ratings.
*   Heatmaps for correlation matrices.

## Next Steps / Hypotheses

Based on the EDA findings, the following hypotheses and directions were formulated for further analysis and modeling:
*   The significant spike in reviews in late 2025 warrants investigation to identify potential causal events.
*   The polarized rating distributions for AliExpress and Alibaba suggest a focus on understanding the drivers of both extreme satisfaction and extreme dissatisfaction for these platforms.
*   Jiji's more balanced rating and higher average sentiment indicate it might be perceived more positively overall; understanding the specific aspects driving this perception is valuable.
*   The topics identified (shipping for AliExpress, price/service for Alibaba, platform usability for Jiji) can guide feature engineering and topic modeling in the machine learning phase.
*   The moderate correlation between VADER scores and star ratings indicates that text-based sentiment analysis can provide additional insights beyond simple numerical ratings, justifying the use of NLP models for sentiment classification.
