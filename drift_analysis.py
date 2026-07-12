import os
import numpy as np
import pandas as pd

def load_aspect_sentiment(path="data/processed/aspect_sentiment.csv"):
    """
    Loads the aspect sentiment CSV, parses year_month into a proper datetime,
    and derives a season column.
    
    Parameters:
    path (str): Path to aspect_sentiment.csv.
    
    Returns:
    pd.DataFrame: Loaded DataFrame with datetime and season columns.
    """
    print(f"Loading aspect sentiment data from: {path}")
    df = pd.read_csv(path)
    
    df['year_month_dt'] = pd.to_datetime(df['year_month'] + '-01')
    
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    df['season'] = df['year_month_dt'].dt.month.map(season_map)
    
    return df

def compute_trend(df, slope_threshold=0.01):
    """
    For each (hotel_id, aspect) group, fits a simple linear regression of 
    avg_sentiment_score against a chronological month index.
    The regression is weighted by review_count (using sqrt(review_count) 
    in np.polyfit to make the WLS objective function weights proportional 
    to review_count).
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing year_month_dt, avg_sentiment_score, review_count.
    slope_threshold (float): Limit to classify direction (improving, declining, stable).
    
    Returns:
    pd.DataFrame: DataFrame containing columns [hotel_id, aspect, trend_slope, trend_direction].
    """
    records = []
    
    for (hotel_id, aspect), group in df.groupby(['hotel_id', 'aspect']):
        if len(group) < 2:
            slope = 0.0
        else:
            group = group.sort_values('year_month_dt')
            min_date = group['year_month_dt'].min()
            
            x = ((group['year_month_dt'].dt.year - min_date.year) * 12 + 
                 (group['year_month_dt'].dt.month - min_date.month)).values
            y = group['avg_sentiment_score'].values
            review_counts = group['review_count'].values
            
            w = np.sqrt(review_counts)
            w = np.clip(w, 1e-5, None)
            
            try:
                slope, intercept = np.polyfit(x, y, deg=1, w=w)
            except Exception:
                try:
                    slope, intercept = np.polyfit(x, y, deg=1)
                except Exception:
                    slope = 0.0
                    
        if slope > slope_threshold:
            direction = "improving"
        elif slope < -slope_threshold:
            direction = "declining"
        else:
            direction = "stable"
            
        records.append({
            'hotel_id': hotel_id,
            'aspect': aspect,
            'trend_slope': slope,
            'trend_direction': direction
        })
        
    return pd.DataFrame(records)

def compute_seasonal_deviation(df):
    """
    For each (hotel_id, aspect) group, computes the average avg_sentiment_score per season.
    Flags a season as 'anomalous' if its average score deviates from the other three 
    seasons' combined average by more than 1 standard deviation (computed across the other 3).
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    pd.DataFrame: Seasonal deviations.
    """
    seasonal_avg = df.groupby(['hotel_id', 'aspect', 'season'])['avg_sentiment_score'].mean().reset_index()
    seasonal_avg.rename(columns={'avg_sentiment_score': 'season_avg_score'}, inplace=True)
    
    records = []
    
    for (hotel_id, aspect), group in seasonal_avg.groupby(['hotel_id', 'aspect']):
        seasons_present = group['season'].tolist()
        scores_present = group['season_avg_score'].tolist()
        season_to_score = dict(zip(seasons_present, scores_present))
        
        for season in seasons_present:
            score = season_to_score[season]
            other_scores = [season_to_score[s] for s in seasons_present if s != season]
            
            if len(other_scores) >= 2:
                other_mean = np.mean(other_scores)
                other_std = np.std(other_scores, ddof=1)
                dev_mag = abs(score - other_mean)
                
                is_anomalous = (dev_mag > other_std) if other_std > 0 else False
            else:
                other_mean = np.mean(other_scores) if other_scores else 0.0
                dev_mag = abs(score - other_mean) if other_scores else 0.0
                is_anomalous = False
                
            records.append({
                'hotel_id': hotel_id,
                'aspect': aspect,
                'season': season,
                'season_avg_score': score,
                'is_anomalous_season': is_anomalous,
                'deviation_magnitude': dev_mag
            })
            
    return pd.DataFrame(records)

def generate_performance_summary(trend_df, seasonal_df):
    """
    Merges trend and seasonal analysis dataframes into a single hotel+aspect summary table.
    If multiple seasons are anomalous, selects the one with the maximum deviation magnitude.
    Saves the output to data/processed/hotel_performance_summary.csv.
    
    Parameters:
    trend_df (pd.DataFrame): Trend results.
    seasonal_df (pd.DataFrame): Seasonal deviation results.
    
    Returns:
    pd.DataFrame: Merged summary DataFrame.
    """
    anomalous = seasonal_df[seasonal_df['is_anomalous_season'] == True]
    
    if not anomalous.empty:
        idx_max = anomalous.groupby(['hotel_id', 'aspect'])['deviation_magnitude'].idxmax()
        flagged_seasons = anomalous.loc[idx_max, ['hotel_id', 'aspect', 'season', 'deviation_magnitude']]
        flagged_seasons.rename(columns={'season': 'flagged_season'}, inplace=True)
    else:
        flagged_seasons = pd.DataFrame(columns=['hotel_id', 'aspect', 'flagged_season', 'deviation_magnitude'])
        
    summary = pd.merge(trend_df, flagged_seasons, on=['hotel_id', 'aspect'], how='left')
    
    summary['flagged_season'] = summary['flagged_season'].where(summary['flagged_season'].notna(), None)
    summary['deviation_magnitude'] = summary['deviation_magnitude'].fillna(0.0)
    
    summary = summary[[
        'hotel_id', 'aspect', 'trend_direction', 'trend_slope', 
        'flagged_season', 'deviation_magnitude'
    ]]
    
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/hotel_performance_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Saved hotel performance summary to: {output_path}")
    
    return summary

def plot_sample_trend(df, hotel_id, aspect):
    """
    Generates a simple time-series line plot of avg_sentiment_score over year_month
    for a given hotel-aspect pair and saves it as a PNG.
    
    Parameters:
    df (pd.DataFrame): Aspect sentiment DataFrame containing year_month_dt, avg_sentiment_score, review_count.
    hotel_id (str): Hotel identifier.
    aspect (str): Aspect name.
    """
    import matplotlib.pyplot as plt
    
    sub_df = df[(df['hotel_id'] == hotel_id) & (df['aspect'] == aspect)].copy()
    if sub_df.empty:
        print(f"No data found for hotel {hotel_id} and aspect {aspect}. Skipping plot.")
        return
    
    sub_df = sub_df.sort_values('year_month_dt')
    
    plt.figure(figsize=(10, 6))
    
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5, zorder=1)
    
    plt.plot(sub_df['year_month'], sub_df['avg_sentiment_score'], marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Avg Sentiment Score', zorder=2)
    
    sizes = np.clip(sub_df['review_count'] * 15, 30, 300)
    plt.scatter(sub_df['year_month'], sub_df['avg_sentiment_score'], s=sizes, color='#ff7f0e', alpha=0.8, zorder=3, label='Review Count (size)')
    
    plt.title(f"Sentiment Trend for Hotel {hotel_id} - Aspect: {aspect.capitalize()}", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Period (Year-Month)", fontsize=12, labelpad=10)
    plt.ylabel("Avg Sentiment Score (Signed)", fontsize=12, labelpad=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(-1.1, 1.1)
    
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    
    plot_dir = "data/processed/plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{hotel_id}_{aspect}_trend.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved trend plot to: {plot_path}")

def main():
    import time
    start_time = time.time()
    
    print("="*60)
    print("Starting Temporal & Seasonal Sentiment Drift Analysis")
    print(f"Project Root: {os.getcwd()}")
    print("="*60)
    
    path = "data/processed/aspect_sentiment.csv"
    if not os.path.exists(path):
        print(f"Error: Input file '{path}' does not exist. Please run pipeline.py first.")
        return
    
    df = load_aspect_sentiment(path)
    print(f"Loaded aspect sentiment data. Shape: {df.shape}")
    
    print("\n" + "-"*40)
    print("Computing sentiment trends (weighted by review count)...")
    trend_df = compute_trend(df, slope_threshold=0.01)
    print(f"Trends computed. Shape: {trend_df.shape}")
    
    trend_counts = trend_df['trend_direction'].value_counts()
    print("\nTrend Direction Distribution:")
    for direction, count in trend_counts.items():
        print(f"  {direction.capitalize()}: {count} pairs")
        
    print("\n" + "-"*40)
    print("Computing seasonal deviations...")
    seasonal_df = compute_seasonal_deviation(df)
    print(f"Seasonal deviations computed. Shape: {seasonal_df.shape}")
    
    total_anomalies = seasonal_df['is_anomalous_season'].sum()
    unique_anomaly_pairs = seasonal_df[seasonal_df['is_anomalous_season'] == True][['hotel_id', 'aspect']].drop_duplicates().shape[0]
    print(f"\nSeasonal Anomaly Stats:")
    print(f"  Total anomalous seasons flagged: {total_anomalies}")
    print(f"  Hotel-aspect pairs with at least one anomalous season: {unique_anomaly_pairs} / {trend_df.shape[0]}")
    
    print("\n" + "-"*40)
    print("Generating performance summary table...")
    summary_df = generate_performance_summary(trend_df, seasonal_df)
    
    print("\n" + "-"*40)
    print("Generating trend plots...")
    declining_pairs = trend_df[trend_df['trend_direction'] == 'declining']
    if not declining_pairs.empty:
        
        target = declining_pairs.sort_values('trend_slope').iloc[0]
        print(f"Selected declining trend for plotting: Hotel {target['hotel_id']}, Aspect {target['aspect']} (slope: {target['trend_slope']:.6f})")
        plot_sample_trend(df, target['hotel_id'], target['aspect'])
    else:
        target = trend_df.sort_values('trend_slope').iloc[0]
        print(f"No declining trends found. Selected lowest slope overall for plotting: Hotel {target['hotel_id']}, Aspect {target['aspect']} (slope: {target['trend_slope']:.6f})")
        plot_sample_trend(df, target['hotel_id'], target['aspect'])
        
    total_time = time.time() - start_time
    print("="*60)
    print(f"Drift analysis completed successfully in {total_time:.2f} seconds.")
    print("="*60)

if __name__ == "__main__":
    main()
