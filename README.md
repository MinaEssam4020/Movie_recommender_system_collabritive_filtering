# Book Recommendation System - Collaborative Filtering

A machine learning project that predicts book ratings and generates personalized recommendations using collaborative filtering.

## About This Project

This project builds a **user-based collaborative filtering** recommender system that:
- Analyzes how similar users rate books
- Predicts ratings for books a user hasn't read yet
- Generates top-N book recommendations based on similar users' preferences

**Dataset:** GoodBooks ratings data with thousands of user-book interactions

## Results

**Model Performance:**
- **Mean Absolute Error (MAE): 0.584 stars**
- ~88% accuracy on predicting user ratings (on a 1-5 scale)
- Tested on 50 random user-book pairs

**What this means:** When the model predicts a 4-star rating, the actual rating is typically between 3.4 and 4.6 stars.

## How It Works

1. **Build Utility Matrix** - Create a user-book rating matrix
2. **Normalize Ratings** - Remove rating bias between different users
3. **Compute Similarity** - Use cosine similarity to find similar users
4. **Predict Ratings** - Weighted average of k similar users' ratings
5. **Generate Recommendations** - Find top books liked by similar users

## Quick Start

```python
# Load the pre-computed data
import pickle
with open('collaborative_filter_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Get top 10 recommendations for user 314
recommendations = collab_generate_top_N_recommendations(user=314, N=10)

# Predict rating for a specific book
predicted_rating = collab_generate_rating_estimate(
    book_title="The Lord of the Rings", 
    user=314
)
```

## Requirements

- Python
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- seaborn

## Key Files

- `Main.ipynb` - Full implementation and analysis
- `book_ratings.csv` - Dataset

## Key Learnings

- Collaborative filtering works well for rating prediction
- User-user similarity is powerful for personalization
- Sparse matrix operations are essential for handling large datasets
- Data normalization significantly improves model performance

## Potential Improvements

- Implement hybrid approach (combine content + collaborative filtering)
- Tune hyperparameters (k neighbors, similarity threshold)
- Handle cold-start problem for new users
- Add item-based collaborative filtering
- Implement matrix factorization (SVD) for scalability

