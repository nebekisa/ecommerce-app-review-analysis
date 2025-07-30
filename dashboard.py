# dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
import base64 # For potential image embedding (like logos)

# --- 1. Configure the Dashboard ---
st.set_page_config(
    page_title="E-commerce App Reviews Dashboard",
    page_icon=":bar_chart:", # Add a relevant icon
    layout="wide", # Use wide layout for better visualization space
    initial_sidebar_state="expanded" # Sidebar expanded by default
)

# Apply custom CSS for styling (Optional but makes it look more professional)
st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e0e0e0;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4F8BF9 !important;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 2. Load Data ---
# Use the cleaned data saved from EDA
# Make sure the path is correct relative to where you run 'streamlit run dashboard.py'
DATA_FILE_PATH = 'cleaned_reviews_data.csv' # Update if needed

@st.cache_data # Cache data loading to improve performance
def load_data():
    """Loads the cleaned review data."""
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        # Ensure ReviewDate is datetime if needed for time series
        df['ReviewDate'] = pd.to_datetime(df['ReviewDate'])
        # Extract Year-Month for grouping if needed
        df['YearMonth'] = df['ReviewDate'].dt.to_period('M').dt.to_timestamp() # Convert for Plotly compatibility
        # Ensure VADER_Sentiment is a categorical type for better sorting/grouping
        df['VADER_Sentiment'] = pd.Categorical(df['VADER_Sentiment'], categories=['Negative', 'Neutral', 'Positive'], ordered=True)
        st.success(f"Data loaded successfully from {DATA_FILE_PATH}")
        return df
    except FileNotFoundError:
        st.error(f"Error: Could not find the data file '{DATA_FILE_PATH}'. Please check the file path.")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        st.error(f"An error occurred while loading the  {e}")
        return pd.DataFrame()

# Load the data
df = load_data()

# Check if data loaded successfully
if df.empty:
    st.stop() # Stop the app if data failed to load

# --- 3. Dashboard Title & Sidebar ---
# Add a title and a brief description
st.title("📱 E-commerce App Review Analysis Dashboard")
st.markdown("Explore customer sentiment and key insights from Google Play Store reviews for popular e-commerce apps.")

# Sidebar for app selection and potentially other filters
st.sidebar.header("Filter Options")

# App Selection
selected_app = st.sidebar.selectbox(
    "Select an E-commerce App:",
    options=df['AppName'].unique(),
    index=0 # Default to the first app
)

# --- 4. Filter Data based on selected app ---
filtered_df = df[df['AppName'] == selected_app].copy() # Use .copy() to avoid SettingWithCopyWarning later

# --- 5. Main Dashboard Content ---
# --- Section 1: Overview Statistics ---
st.header(f"Overview for {selected_app}", divider='blue') # Add a divider for visual separation

# Use columns for a more structured layout
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Reviews", value=filtered_df.shape[0])
with col2:
    avg_rating = filtered_df['Rating'].mean()
    st.metric(label="Average Rating", value=f"{avg_rating:.2f} ⭐")
with col3:
    # Overall sentiment (e.g., percentage of Positive)
    # --- FIX 1: Add observed=True to groupby to silence deprecation warning ---
    sentiment_counts = filtered_df.groupby('VADER_Sentiment', observed=True).size()
    positive_pct = (sentiment_counts.get('Positive', 0) / len(filtered_df)) * 100
    st.metric(label="Positive Sentiment (%)", value=f"{positive_pct:.1f}%")
with col4:
    # Example: Dominant Sentiment Category
    dominant_sentiment = sentiment_counts.idxmax() if not sentiment_counts.empty else "N/A"
    st.metric(label="Dominant Sentiment", value=dominant_sentiment)

# --- Section 2: Visualizations ---
# --- 2.1 Rating Distribution & Sentiment Distribution Side-by-Side ---
st.subheader("Distributions", divider='gray')
col_dist1, col_dist2 = st.columns(2)

with col_dist1:
    # Rating Distribution
    st.write("**Rating Distribution (1-5 Stars)**")
    rating_counts = filtered_df['Rating'].value_counts().sort_index()
    fig_rating = px.bar(x=rating_counts.index, y=rating_counts.values,
                        labels={'x':'Rating (Stars)', 'y':'Number of Reviews'},
                        title="", # Title handled by markdown above
                        color=rating_counts.index, # Color bars by rating
                        color_continuous_scale='Bluered_r') # Red for low ratings, Blue for high
    fig_rating.update_layout(showlegend=False, height=400) # Adjust height
    st.plotly_chart(fig_rating, use_container_width=True, theme="streamlit")

with col_dist2:
    # Sentiment Distribution
    st.write("**Sentiment Distribution (VADER)**")
    # Sort index to ensure Negative, Neutral, Positive order
    # --- FIX 1: Add observed=True here as well ---
    sentiment_counts_sorted = sentiment_counts.reindex(['Negative', 'Neutral', 'Positive'], fill_value=0)
    fig_sentiment = px.pie(names=sentiment_counts_sorted.index, values=sentiment_counts_sorted.values,
                           title="", # Title handled by markdown above
                           color_discrete_sequence=px.colors.qualitative.Set1) # Use a distinct color set
    fig_sentiment.update_layout(height=400) # Adjust height
    st.plotly_chart(fig_sentiment, use_container_width=True, theme="streamlit")


# --- 2.2 Sentiment Trend Over Time (Monthly) ---
st.subheader("Sentiment Trend Over Time (Monthly)", divider='gray')

# Group by YearMonth and Sentiment
# --- FIX 1: Add observed=True to this groupby as well ---
sentiment_trend = filtered_df.groupby(['YearMonth', 'VADER_Sentiment'], observed=True).size().reset_index(name='Count')
# Ensure YearMonth is datetime for sorting
sentiment_trend['YearMonth'] = pd.to_datetime(sentiment_trend['YearMonth'])
# Sort by date
sentiment_trend = sentiment_trend.sort_values('YearMonth')

# Create a line chart for sentiment trend
# --- FIX 2: Ensure the figure variable name is consistent (fig_sentiment_trend) ---
fig_sentiment_trend = px.line(sentiment_trend, x='YearMonth', y='Count', color='VADER_Sentiment',
                              title=f"Monthly Sentiment Trend for {selected_app}",
                              labels={'YearMonth':'Date', 'Count':'Number of Reviews'},
                              markers=True, # Add markers to points
                              color_discrete_map={'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'}) # Custom colors

fig_sentiment_trend.update_xaxes(title_text='Date')
fig_sentiment_trend.update_yaxes(title_text='Number of Reviews')
fig_sentiment_trend.update_layout(hovermode='x unified', height=500) # Unified hover, adjust height
# --- FIX 2: Use the correct figure variable name (fig_sentiment_trend) ---
st.plotly_chart(fig_sentiment_trend, use_container_width=True, theme="streamlit")


# --- 2.3 Top Words/N-grams Analysis ---
st.subheader("Top Words & Phrases", divider='gray')
# Tabs for different analysis levels (Unigrams, Bigrams, Word Cloud)
tab_words1, tab_words2, tab_words3 = st.tabs(["Top Unigrams", "Top Bigrams", "Word Cloud"])

with tab_words1:
    st.write("**Most Frequent Single Words**")
    # Combine all text for the selected app
    all_text_uni = " ".join(filtered_df['CleanedReviewText'].dropna().tolist())
    if all_text_uni.strip():
        vectorizer_uni = CountVectorizer(max_features=20, stop_words='english', ngram_range=(1,1))
        try:
            X_uni = vectorizer_uni.fit_transform([all_text_uni])
            words_uni = vectorizer_uni.get_feature_names_out()
            counts_uni = np.array(X_uni.sum(axis=0)).flatten()
            word_df_uni = pd.DataFrame({'Word': words_uni, 'Count': counts_uni}).sort_values(by='Count', ascending=False)
            fig_words_uni = px.bar(word_df_uni, x='Count', y='Word', orientation='h',
                                   title="", labels={'Count':'Frequency', 'Word':'Word'},
                                   color='Count', color_continuous_scale='Viridis')
            fig_words_uni.update_layout(height=500)
            st.plotly_chart(fig_words_uni, use_container_width=True, theme="streamlit")
        except ValueError:
            st.warning("Not enough text data to generate unigram frequencies.")
    else:
        st.warning("No text data available for unigram analysis.")

with tab_words2:
    st.write("**Most Frequent Two-Word Phrases**")
    # Combine all text for the selected app
    all_text_bi = " ".join(filtered_df['CleanedReviewText'].dropna().tolist())
    if all_text_bi.strip():
        vectorizer_bi = CountVectorizer(max_features=20, stop_words='english', ngram_range=(2,2))
        try:
            X_bi = vectorizer_bi.fit_transform([all_text_bi])
            words_bi = vectorizer_bi.get_feature_names_out()
            counts_bi = np.array(X_bi.sum(axis=0)).flatten()
            word_df_bi = pd.DataFrame({'Phrase': words_bi, 'Count': counts_bi}).sort_values(by='Count', ascending=False)
            fig_words_bi = px.bar(word_df_bi, x='Count', y='Phrase', orientation='h',
                                  title="", labels={'Count':'Frequency', 'Phrase':'Phrase'},
                                  color='Count', color_continuous_scale='Plasma')
            fig_words_bi.update_layout(height=500)
            st.plotly_chart(fig_words_bi, use_container_width=True, theme="streamlit")
        except ValueError:
            st.warning("Not enough text data to generate bigram frequencies.")
    else:
        st.warning("No text data available for bigram analysis.")

with tab_words3:
    st.write("**Word Cloud Visualization**")
    # Combine all text for the selected app
    all_text_wc = " ".join(filtered_df['CleanedReviewText'].dropna().tolist())
    if all_text_wc.strip():
        try:
            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, colormap='viridis').generate(all_text_wc)
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc)
        except Exception as e:
             st.error(f"Error generating word cloud: {e}")
    else:
        st.warning("No text data available for word cloud generation.")


# --- 2.4 Example Reviews ---
st.subheader("Sample Reviews by Sentiment", divider='gray')
# Show examples for each sentiment category using tabs
sentiments = ['Positive', 'Negative', 'Neutral']
tabs_reviews = st.tabs([f"😊 {s}" for s in sentiments]) # Add emojis to tabs

for i, sentiment in enumerate(sentiments):
    with tabs_reviews[i]:
        st.write(f"**Sample {sentiment} Reviews**")
        # Filter for the specific sentiment within the selected app
        # --- FIX 1: Add observed=True here if filtering/grouping by sentiment later ---
        # (Not strictly necessary here as we are filtering directly, but good practice if groupby is used)
        sentiment_reviews = filtered_df[filtered_df['VADER_Sentiment'] == sentiment]
        if not sentiment_reviews.empty:
            # Example strategy: Show ones with extreme VADER scores for impact, or just a sample
            if sentiment == 'Positive':
                examples = sentiment_reviews.nlargest(5, 'VADER_Score') # Top 5 most positive
            elif sentiment == 'Negative':
                examples = sentiment_reviews.nsmallest(5, 'VADER_Score') # Top 5 most negative
            else: # Neutral
                # For neutral, just take a sample
                examples = sentiment_reviews.sample(n=min(5, len(sentiment_reviews)), random_state=42)

            # Display examples
            for idx, row in examples.iterrows():
                 # Use expander for cleaner look
                 with st.expander(f"Rating: {row['Rating']} | VADER Score: {row['VADER_Score']:.3f}", expanded=False):
                     st.write(f"**Review Text:** {row['ReviewText']}")
                     # Optional: Add more details if available and relevant
                     # st.write(f"Date: {row['ReviewDate'].strftime('%Y-%m-%d')}")
                     # st.write(f"Review ID: {row.get('PlayStoreReviewID', 'N/A')}")
        else:
            st.write(f"No {sentiment.lower()} reviews found for {selected_app}.")

# --- 8. Footer ---
st.markdown("---")
# Get current year dynamically
current_year = datetime.now().year
st.caption(f"Dashboard built with Streamlit. Data sourced from Google Play Store reviews. © {current_year}")
