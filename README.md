# Hotel Review Intelligence Engine

**Expedia Group Campus Hackathon 2026 — Innovation Round**
**Problem Statement 2: Hotel Review Intelligence Engine**

## Overview

This project analyzes hotel reviews and user profiles to (1) track hotel
performance over time, (2) detect aspect-level and seasonal sentiment shifts,
and (3) generate personalized, evidence-based hotel recommendations.

## Architecture / Pipeline
1. **Inputs:** `data/hotel_reviews.json`, `data/user_profiles.json`
2. **`pipeline.py`** → cleans data, tags aspects, runs sentiment analysis →
   outputs `data/processed/aspect_sentiment.csv`
3. **`drift_analysis.py`** → detects trend/seasonal drift from that CSV →
   outputs `data/processed/hotel_performance_summary.csv` and
   `data/processed/plots/*.png`
4. **`recommend.py`** → builds per-hotel aspect scores, matches them against
   each user profile → outputs `data/processed/recommendations.json`

### 1. `pipeline.py` — Data Loading, Cleaning & Aspect Sentiment
- Loads hotel reviews and user profiles.
- Cleans review text, parses dates into year/month/quarter/season.
- Splits reviews into sentences, maps sentences to 5 aspects (cleanliness,
  service, amenities, value, location) via keyword matching.
- Runs local sentiment inference (Hugging Face
  `distilbert-base-uncased-finetuned-sst-2-english`, PyTorch backend) on each
  aspect-tagged sentence, converting model output into a **signed score**
  (positive label → +confidence, negative label → -confidence) so opposing
  sentiments cancel rather than compound.
- Sentences matching multiple aspects are tagged under all matching aspects.
- Aggregates to `hotel_id | aspect | year_month | avg_sentiment_score | review_count`,
  with `review_count` deduplicated by unique `review_id`.
- Sentence-level deduplication before inference gives a large speedup (see
  "Key Dataset Finding" below).

### 2. `drift_analysis.py` — Temporal & Seasonal Drift Detection
- Fits a simple linear trend (slope) of sentiment over time per hotel+aspect,
  classified as improving / declining / stable against a configurable
  threshold.
- Flags a season as "anomalous" for a hotel+aspect if its average deviates
  by more than 1 standard deviation from the other three seasons.
- Outputs `hotel_id | aspect | trend_direction | trend_slope | flagged_season | deviation_magnitude`.
- Generates sample time-series plots (see `data/processed/plots/`).

### 3. `recommend.py` — Personalization & Recommendations
- Parses each free-text user profile description into:
  - `desired_dims`: fine-grained interest tags (e.g. `safety`, `local_culture`,
    `business_connectivity`) inferred via keyword matching.
  - `archetype`: a short descriptive slug derived from persona + dimension cues.
- Builds a per-hotel aspect score (review-count-weighted average across time).
- Scores each hotel per profile by weighting aspect scores against the
  profile's inferred aspect priorities.
- Returns the **top 5 hotels per profile**, each with a `relevance_score` and
  a human-readable `evidence` string naming the strongest positive-contributing
  aspects (and a caveat if a heavily-weighted aspect is strongly negative).
- Ties are broken using total review volume as a confidence proxy.

## Key Dataset Findings (Important Context)

1. **Review text is combinatorially generated from 43 fixed sentence
   templates.** Out of 50,000 reviews, only 28,411 are unique at the full-text
   level, and all aspect-relevant sentences resolve to just 43 recurring
   templates. We exploited this for a ~1,470x inference speedup via
   sentence-level deduplication before running the sentiment model. This also
   means sentiment scores are deterministic per template — trend/seasonal
   "drift" in the sample data reflects which templates were assigned to which
   months for a given hotel, not organic sentiment change. Our pipeline logic
   is nonetheless built to generalize correctly to organically-written review
   text.
2. **User profile descriptions also show a repeated-template pattern** —
   several of the 50 profiles produce identical inferred archetypes and
   recommendations, consistent with the same combinatorial generation
   approach used for reviews. Our system correctly produces deterministic,
   consistent output for repeated personas, which is a desirable property in
   a production recommender.
3. **Persona–template alignment**: review templates cluster thematically
   (business/WiFi, family/pool, solo-safety, etc.) in a way that visibly
   aligns with the traveler personas in `user_profiles.json`, which directly
   informed our personalization design (Step 6/7).

## Setup Instructions

```bash
git clone https://github.com/AayuSiuu/expedia.git
cd expedia
pip install -r requirements.txt
python pipeline.py
python drift_analysis.py
python recommend.py
```

Requires Python 3.10+. All paths are relative to the project root.

## Assumptions

- Where the sample output schema image was partially cut off, we retained
  additional fields (`relevance_score`, `evidence`) beyond what was directly
  visible, as they materially support the "evidence-based" requirement in the
  problem statement. Field names (`profile_id`, `archetype`, `desired_dims`,
  `top_hotels`) match the visible portion of the provided schema exactly.
- Nulls in review text are treated as empty strings rather than dropped, to
  preserve row alignment with `review_id`.
- Sentences matching no aspect keyword are dropped by default (`drop_unhandled=True`),
  configurable to a "general" bucket instead.
- A binary sentiment model (positive/negative, no neutral class) was used for
  speed; this is a known simplification (see Limitations).

## Limitations

- Sentiment model has no neutral class — mixed/neutral sentences are forced
  into positive or negative.
- Contradiction handling across reviewer types (Step 5) was not implemented
  in this submission — see Future Improvements.
- Relevance scores can cluster near the ceiling (±1) due to the underlying
  sentiment model producing high-confidence scores on the dataset's templated
  sentences; ties are broken using review volume as a proxy for confidence.
- No RAG / vector retrieval was used — not required per the problem statement
  FAQ, and keyword-based aspect/dimension matching was sufficient and more
  explainable given the dataset's template structure.

## Future Improvements

- Explicit contradiction handling across reviewer types (Step 5).
- Zero-shot aspect classification (e.g. `facebook/bart-large-mnli`) to remove
  dependency on hand-written keyword lists.
- A neutral-aware sentiment model for more nuanced scoring.
- Semantic retrieval (sentence-transformers + FAISS/Chroma) as an alternative
  or complement to keyword-based profile matching, particularly useful if
  profile descriptions become more varied/organic than the current dataset.

## Tools Used

Python, Pandas, NumPy, Hugging Face Transformers (PyTorch backend), Matplotlib,
tqdm.
