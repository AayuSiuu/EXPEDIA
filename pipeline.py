import os
import re
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import pipeline

def load_data(reviews_path="data/hotel_reviews.json", profiles_path="data/user_profiles.json"):
    """
    Loads hotel reviews and user profiles JSON files.
    Prints schema, shape, null counts, and rating distribution for hotel_reviews.
    
    Parameters:
    reviews_path (str): Path to hotel reviews JSON file.
    profiles_path (str): Path to user profiles JSON file.
    
    Returns:
    tuple: (reviews_df, profiles_df)
    """
    print(f"Loading reviews from: {reviews_path}")
    reviews_df = pd.read_json(reviews_path)
    print(f"Loading user profiles from: {profiles_path}")
    profiles_df = pd.read_json(profiles_path)
    
    print("\n--- Hotel Reviews Dataset Stats ---")
    print(f"Shape: {reviews_df.shape}")
    print("\nSchema (dtypes):")
    print(reviews_df.dtypes)
    print("\nNull Counts:")
    print(reviews_df.isnull().sum())
    
    if 'rating' in reviews_df.columns:
        print("\nRating Distribution:")
        rating_dist = reviews_df['rating'].value_counts().sort_index()
        rating_pct = reviews_df['rating'].value_counts(normalize=True).sort_index() * 100
        for rating, pct in zip(rating_dist.index, rating_pct.values):
            bar = '#' * int(pct // 2)
            print(f"  Rating {rating}: {pct:5.1f}% | {bar}")
            
    print("-----------------------------------")
    
    return reviews_df, profiles_df

def clean_and_parse_data(df):
    """
    Cleans review text (lowercase, strip special characters, handle nulls)
    and parses the review date into year, month, quarter, season, and year_month columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing hotel reviews.
    
    Returns:
    pd.DataFrame: Cleaned and parsed DataFrame.
    """
    df = df.copy()
    
    df['review_text'] = df['review_text'].fillna("").astype(str)
    
    def clean_text(text):
        cleaned = text.lower()
        cleaned = re.sub(r"[^a-z0-9\s\.\!\?\,\'\-]", "", cleaned)
        # normalize multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
        
    df['cleaned_review_text'] = df['review_text'].apply(clean_text)
    
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    
    df['year'] = df['review_date'].dt.year
    df['month'] = df['review_date'].dt.month
    df['quarter'] = df['review_date'].dt.quarter
    
    # Season mapping
    # Winter: Dec (12), Jan (1), Feb (2)
    # Spring: Mar (3), Apr (4), May (5)
    # Summer: Jun (6), Jul (7), Aug (8)
    # Autumn: Sep (9), Oct (10), Nov (11)
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    df['season'] = df['month'].map(season_map)
    df['year_month'] = df['review_date'].dt.strftime('%Y-%m')
    
    # Fill potential null values from coercion
    df['year'] = df['year'].fillna(-1).astype(int)
    df['month'] = df['month'].fillna(-1).astype(int)
    df['quarter'] = df['quarter'].fillna(-1).astype(int)
    df['season'] = df['season'].fillna("Unknown")
    df['year_month'] = df['year_month'].fillna("Unknown")
    
    return df

def extract_aspects_and_sentiment(df, sentiment_pipeline, drop_unhandled=True):
    """
    Implements aspect-based sentiment tagging. Splits reviews into sentences,
    assigns aspects using keywords, runs local Hugging Face sentiment model,
    and returns a long-format DataFrame with sentiment scores.
    
    Parameters:
    df (pd.DataFrame): Cleaned and parsed reviews DataFrame.
    sentiment_pipeline: Hugging Face sentiment analysis pipeline.
    drop_unhandled (bool): If True, drops sentences matching no aspect. If False, tags them as 'general'.
    
    Returns:
    pd.DataFrame: Long-format DataFrame with columns [review_id, hotel_id, aspect, year_month, sentiment_score, sentence].
    """
    aspect_patterns = {
        'cleanliness': re.compile(r'\b(clean|dirty|dusty|spotless|hygienic|smell|neat|messy|stain|wash|sheets|bathroom)s?\b', re.IGNORECASE),
        'service': re.compile(r'\b(service|staff|friendly|helpful|rude|check-in|check-out|desk|manager|hospitality)s?\b', re.IGNORECASE),
        'amenities': re.compile(r'\b(pool|gym|wifi|internet|breakfast|parking|elevator|ac|heater|tv|beds?|rooms?|showers?)\b', re.IGNORECASE),
        'value': re.compile(r'\b(price|cost|expensive|cheap|value|worth|money|affordable|overpriced|deals?)\b', re.IGNORECASE),
        'location': re.compile(r'\b(location|near|close|walk|distance|view|noisy|street|subway|airport|downtown|area)s?\b', re.IGNORECASE)
    }
    
    records = []
    
    print("Splitting reviews into sentences and matching aspects...")
    for idx, row in df.iterrows():
        review_id = row['review_id']
        hotel_id = row['hotel_id']
        year_month = row['year_month']
        text = row['cleaned_review_text']
        
        sentences = [s.strip() for s in re.split(r'(?<!\bmr\.)(?<!\bdr\.)(?<!\bms\.)(?<!\bvs\.)(?<!\bst\.)(?<!\betc\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<=[.!?])\s+', text, flags=re.IGNORECASE) if s.strip()]
        
        if not sentences and text.strip():
            sentences = [text.strip()]
            
        for sentence in sentences:
            matched_aspects = []
            for aspect, pattern in aspect_patterns.items():
                if pattern.search(sentence):
                    matched_aspects.append(aspect)
                    
            if matched_aspects:
               for aspect in matched_aspects:
                    records.append({
                        'review_id': review_id,
                        'hotel_id': hotel_id,
                        'year_month': year_month,
                        'aspect': aspect,
                        'sentence': sentence
                    })
            else:
                if not drop_unhandled:
                    records.append({
                        'review_id': review_id,
                        'hotel_id': hotel_id,
                        'year_month': year_month,
                        'aspect': 'general',
                        'sentence': sentence
                    })
                    
    if not records:
        print("No aspect-matched sentences found.")
        return pd.DataFrame(columns=['review_id', 'hotel_id', 'aspect', 'year_month', 'sentiment_score', 'sentence'])
        
    tagged_df = pd.DataFrame(records)
    print(f"Total aspect-sentence records extracted: {len(tagged_df)}")
    
    unique_sentences = tagged_df['sentence'].unique().tolist()
    print(f"Unique sentences to analyze: {len(unique_sentences)}")
    
    sentence_sentiment = {}
    batch_size = 128
    
    # Progress bar using tqdm
    print("Running Hugging Face sentiment model locally...")
    for i in tqdm(range(0, len(unique_sentences), batch_size), desc="Analyzing sentiment"):
        batch = unique_sentences[i:i+batch_size]
        batch_results = sentiment_pipeline(batch)
        
        for sentence, result in zip(batch, batch_results):
            label = result['label'].upper()
            confidence = result['score']
            # Signed numeric score: Positive label -> +confidence, Negative label -> -confidence
            signed_score = confidence if label == 'POSITIVE' else -confidence
            sentence_sentiment[sentence] = signed_score
            
    # Map signed scores back to the records
    tagged_df['sentiment_score'] = tagged_df['sentence'].map(sentence_sentiment)
    
    return tagged_df

def aggregate_sentiment(tagged_df):
    """
    Aggregates sentiment score and count per hotel_id, aspect, and year_month.
    Ensures review_count counts unique reviews (deduplicated by review_id).
    
    Parameters:
    tagged_df (pd.DataFrame): DataFrame with columns [review_id, hotel_id, aspect, year_month, sentiment_score].
    
    Returns:
    pd.DataFrame: Aggregated DataFrame.
    """
    if tagged_df.empty:
        return pd.DataFrame(columns=['hotel_id', 'aspect', 'year_month', 'avg_sentiment_score', 'review_count'])
        
    # Group by hotel_id, aspect, year_month
    # avg_sentiment_score = mean of sentiment_score
    # review_count = count of unique review_id
    aggregated = tagged_df.groupby(['hotel_id', 'aspect', 'year_month']).agg(
        avg_sentiment_score=('sentiment_score', 'mean'),
        review_count=('review_id', 'nunique')
    ).reset_index()
    
    return aggregated

def save_processed_data(df, output_path="data/processed/aspect_sentiment.csv"):
    """
    Saves the aggregated DataFrame to a CSV file. Creates directory structure if missing.
    
    Parameters:
    df (pd.DataFrame): DataFrame to save.
    output_path (str): File path to save CSV to.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved aggregated output to: {output_path}")

def main(slice_n=None, drop_unhandled=True):
    """
    Main function to orchestrate the pipeline.
    
    Parameters:
    slice_n (int): If provided, runs the pipeline on a slice of the reviews.
    drop_unhandled (bool): Passed to extract_aspects_and_sentiment.
    """
    start_time = time.time()
    
    reviews_path = "data/hotel_reviews.json"
    profiles_path = "data/user_profiles.json"
    output_path = "data/processed/aspect_sentiment.csv"
    
    print("="*60)
    print("Starting Hotel Review Intelligence Engine Pipeline")
    print(f"Project Root: {os.getcwd()}")
    print("="*60)
    
    # Load Data
    reviews_df, profiles_df = load_data(reviews_path, profiles_path)
    
    # Slice if requested
    if slice_n is not None:
        print(f"\nTaking a small slice of {slice_n} reviews for testing...")
        reviews_df = reviews_df.head(slice_n)
        
    print(f"\nReviews shape at start: {reviews_df.shape}")
    print(f"Reviews schema at start:")
    print(reviews_df.dtypes)
    
    # Clean and Parse Data
    print("\n" + "-"*40)
    print("Cleaning and Parsing Data...")
    cleaned_df = clean_and_parse_data(reviews_df)
    print(f"Cleaned DataFrame shape: {cleaned_df.shape}")
    print("Columns added: year, month, quarter, season, year_month, cleaned_review_text")
    
    #Initialize Hugging Face pipeline
    print("\n" + "-"*40)
    print("Initializing Hugging Face Sentiment Analysis Pipeline...")
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english", 
        framework="pt",
        device=device
    )
    
    # Aspect-Based Sentiment Tagging
    print("\n" + "-"*40)
    print(f"Extracting Aspects & Sentiments (drop_unhandled={drop_unhandled})...")
    tagged_df = extract_aspects_and_sentiment(cleaned_df, sentiment_pipeline, drop_unhandled=drop_unhandled)
    print(f"Tagged output shape: {tagged_df.shape}")
    print("Tagged schema:")
    print(tagged_df.dtypes)
    
    # Show sample rows
    if not tagged_df.empty:
        print("\nSample tagged output (first 5 rows before aggregation):")
        print(tagged_df[['review_id', 'hotel_id', 'aspect', 'year_month', 'sentiment_score', 'sentence']].head(5).to_string())
    
    # Aggregate Output
    print("\n" + "-"*40)
    print("Aggregating sentiment scores...")
    aggregated_df = aggregate_sentiment(tagged_df)
    print(f"Aggregated output shape: {aggregated_df.shape}")
    print("Aggregated schema:")
    print(aggregated_df.dtypes)
    
    # Show sample aggregated records
    if not aggregated_df.empty:
        print("\nSample aggregated output (first 10 rows):")
        print(aggregated_df.head(10).to_string())
        
    # Save final output
    print("\n" + "-"*40)
    print("Saving aggregated output...")
    save_processed_data(aggregated_df, output_path)
    
    total_time = time.time() - start_time
    print("="*60)
    print(f"Pipeline completed successfully in {total_time:.2f} seconds.")
    print("="*60)
    
    return cleaned_df, tagged_df, aggregated_df
if __name__ == "__main__":
    main(slice_n=None, drop_unhandled=True)
