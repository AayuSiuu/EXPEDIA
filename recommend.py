import pandas as pd
import re
import json

ASPECT_KEYWORDS = {
    "cleanliness": ["clean", "hygienic", "spotless", "housekeeping", "fresh", "tidy"],
    "service": ["service", "staff", "helpful", "friendly", "hospitality", "front desk", "check-in", "attentive"],
    "amenities": ["wifi", "internet", "pool", "gym", "breakfast", "meeting", "desk", "kids", "connecting room",
                  "accessible", "step-free", "family-friendly", "workspace", "spa"],
    "value": ["budget", "afford", "price", "cost", "expense", "value", "cheap", "rate"],
    "location": ["central", "walk", "location", "safe", "safety", "quiet", "downtown", "culture",
                 "market", "street", "area", "district", "office"],
}

DIMENSION_KEYWORDS = {
    "safety": ["safe", "safety", "secure", "security", "well-lit"],
    "local_culture": ["culture", "local", "market", "artisan", "authentic", "neighborhood"],
    "location_central": ["central", "downtown", "walk", "walkable", "distance", "near", "office district"],
    "business_connectivity": ["wifi", "internet", "meeting", "desk", "workspace", "conference", "call"],
    "family_friendly": ["kids", "toddler", "family", "connecting room", "children", "child"],
    "budget_value": ["budget", "afford", "cheap", "value", "cost", "price", "expense"],
    "luxury_comfort": ["luxury", "indulgent", "refinement", "five-star", "premium"],
    "accessibility": ["accessible", "step-free", "wheelchair", "roll-in"],
    "quiet_relaxation": ["quiet", "peaceful", "relax", "spa", "soundproof"],
    "cleanliness_priority": ["clean", "hygienic", "spotless", "housekeeping"],
}


def load_data():
    profiles = pd.read_json("data/user_profiles.json")
    aspect_sentiment = pd.read_csv("data/processed/aspect_sentiment.csv")
    reviews = pd.read_json("data/hotel_reviews.json")
    return profiles, aspect_sentiment, reviews


def build_hotel_aspect_scores(aspect_sentiment: pd.DataFrame) -> pd.DataFrame:
    def weighted_avg(g):
        return (g["avg_sentiment_score"] * g["review_count"]).sum() / g["review_count"].sum()

    hotel_aspect = (
        aspect_sentiment.groupby(["hotel_id", "aspect"])
        .apply(lambda g: pd.Series({
            "overall_aspect_score": weighted_avg(g),
            "total_reviews": g["review_count"].sum()
        }))
        .reset_index()
    )
    return hotel_aspect


def get_profile_aspect_weights(description: str) -> dict:
    text = description.lower()
    raw_weights = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        raw_weights[aspect] = hits

    total = sum(raw_weights.values())
    if total == 0:
        return {a: 1 / len(ASPECT_KEYWORDS) for a in ASPECT_KEYWORDS}
    return {a: w / total for a, w in raw_weights.items()}


def infer_desired_dims(description: str, max_dims=3) -> list:
    """Extract the top matching fine-grained dimension tags from the free-text
    profile description, based on keyword hits. Returns up to max_dims tags,
    ranked by number of keyword matches."""
    text = description.lower()
    scored_dims = []
    for dim, keywords in DIMENSION_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits > 0:
            scored_dims.append((dim, hits))
    scored_dims.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored_dims[:max_dims]] if scored_dims else ["general"]


def infer_archetype(description: str, desired_dims: list) -> str:
    """Build a short slug archetype from persona cues + top dims, e.g. 
    'solo_female_culture'. Falls back to a generic slug from dims if no 
    persona cue keywords are found."""
    text = description.lower()
    persona_tags = []

    if "solo" in text:
        persona_tags.append("solo")
    if "female" in text:
        persona_tags.append("female")
    if "corporate" in text or "business" in text or "road-warrior" in text or "road warrior" in text:
        persona_tags.append("corporate")
    if "famil" in text or "toddler" in text or "kids" in text or "children" in text:
        persona_tags.append("family")
    if "couple" in text or "honeymoon" in text:
        persona_tags.append("couple")
    if "budget" in text:
        persona_tags.append("budget")

    if persona_tags:
        # combine persona tags with the single most relevant dim for specificity
        top_dim = desired_dims[0] if desired_dims else "general"
        slug_parts = persona_tags + [top_dim.split("_")[0]]
        # dedupe while preserving order
        seen = set()
        slug_parts = [p for p in slug_parts if not (p in seen or seen.add(p))]
        return "_".join(slug_parts[:3])
    else:
        return "_".join(desired_dims[:2]) if desired_dims else "general_traveler"


def score_hotels_for_profile(profile_weights: dict, hotel_aspect_scores: pd.DataFrame) -> pd.DataFrame:
    pivot = hotel_aspect_scores.pivot(index="hotel_id", columns="aspect", values="overall_aspect_score").fillna(0)
    review_counts = hotel_aspect_scores.pivot(index="hotel_id", columns="aspect", values="total_reviews").fillna(0)

    relevance = pd.Series(0.0, index=pivot.index)
    for aspect, weight in profile_weights.items():
        if aspect in pivot.columns:
            relevance += pivot[aspect] * weight

    total_evidence = review_counts.sum(axis=1)

    result = pd.DataFrame({
        "hotel_id": pivot.index,
        "relevance_score": relevance.values,
        "total_reviews": total_evidence.values
    })
    return result.sort_values(["relevance_score", "total_reviews"], ascending=[False, False])


def get_hotel_name(reviews: pd.DataFrame, hotel_id: str) -> str:
    match = reviews.loc[reviews["hotel_id"] == hotel_id, "hotel_name"]
    return match.iloc[0] if not match.empty else hotel_id


def build_evidence(hotel_aspect_scores: pd.DataFrame, hotel_id: str, profile_weights: dict, top_n_aspects=2) -> str:
    hotel_rows = hotel_aspect_scores[hotel_aspect_scores["hotel_id"] == hotel_id]
    weighted = []
    for _, row in hotel_rows.iterrows():
        w = profile_weights.get(row["aspect"], 0)
        weighted.append((row["aspect"], row["overall_aspect_score"], w * row["overall_aspect_score"]))
    weighted.sort(key=lambda x: x[2], reverse=True)

    positives = [w for w in weighted if w[1] > 0][:top_n_aspects]
    negatives = [w for w in weighted if w[1] < -0.5 and w[2] > 0.1]

    parts = []
    if positives:
        parts.append("Strong match on: " + ", ".join(f"{a} ({s:.2f})" for a, s, _ in positives))
    else:
        parts.append("No strongly positive aspects found for this profile's priorities.")
    if negatives:
        parts.append("Caveat — weak on: " + ", ".join(f"{a} ({s:.2f})" for a, s, _ in negatives))

    return " | ".join(parts)


def get_top5_recommendations(profiles, hotel_aspect_scores, reviews, top_n=5):
    all_recommendations = {}
    for _, profile in profiles.iterrows():
        description = profile["description"]
        weights = get_profile_aspect_weights(description)
        desired_dims = infer_desired_dims(description)
        archetype = infer_archetype(description, desired_dims)

        scored = score_hotels_for_profile(weights, hotel_aspect_scores)
        top5 = scored.head(top_n)

        top_hotels = []
        for rank, (_, row) in enumerate(top5.iterrows(), start=1):
            top_hotels.append({
                "rank": rank,
                "hotel_id": row["hotel_id"],
                "hotel_name": get_hotel_name(reviews, row["hotel_id"]),
                "relevance_score": round(row["relevance_score"], 4),
                "evidence": build_evidence(hotel_aspect_scores, row["hotel_id"], weights)
            })

        all_recommendations[profile["profile_id"]] = {
            "profile_id": profile["profile_id"],
            "archetype": archetype,
            "desired_dims": desired_dims,
            "top_hotels": top_hotels
        }
    return all_recommendations


def main():
    profiles, aspect_sentiment, reviews = load_data()
    print(f"Loaded {len(profiles)} profiles, {len(aspect_sentiment)} aspect-sentiment rows")

    hotel_aspect_scores = build_hotel_aspect_scores(aspect_sentiment)
    print(f"Built aspect scores for {hotel_aspect_scores['hotel_id'].nunique()} hotels")

    recommendations = get_top5_recommendations(profiles, hotel_aspect_scores, reviews)

    with open("data/processed/recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2)

    sample_id = list(recommendations.keys())[0]
    print(f"\nSample output for {sample_id}:")
    print(json.dumps(recommendations[sample_id], indent=2))


if __name__ == "__main__":
    main()