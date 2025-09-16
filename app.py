# app.py

import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Base directory and data file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'static', 'data', 'books_clean.csv')

# Global variables
df_clean = None
cosine_sim = None
indices = None
use_collab = False
pop_scores = None
predicted_ratings = None

def load_data():
    global df_clean, cosine_sim, indices, use_collab, pop_scores, predicted_ratings

    # Load books_clean.csv
    try:
        app.logger.info(f"Loading books CSV from: {CSV_PATH}")
        app.logger.info(f"Exists? {os.path.exists(CSV_PATH)}")
        df_clean = pd.read_csv(CSV_PATH)
        app.logger.info(f"Loaded rows: {len(df_clean)}")
    except Exception as e:
        app.logger.error(f"Failed to load books_clean.csv: {e}")
        return False

    # Ensure required columns
    for col in ['title', 'authors_clean', 'categories_clean', 'combined_features']:
        if col not in df_clean.columns:
            app.logger.error(f"Missing column: {col}")
            return False
        df_clean[col] = df_clean[col].fillna('').astype(str)

    # Optional fallback columns
    if 'average_rating_clean' not in df_clean.columns:
        df_clean['average_rating_clean'] = 0.0
    if 'ratings_count_clean' not in df_clean.columns:
        df_clean['ratings_count_clean'] = 0

    # Collaborative filtering model
    try:
        ratings_path = os.path.join(BASE_DIR, 'static', 'data', 'ratings.csv')
        ratings_df = pd.read_csv(ratings_path)
        user_item = ratings_df.pivot_table(
            index='user_id', columns='book_id', values='rating'
        ).fillna(0)
        nmf_model = NMF(n_components=20, random_state=42)
        W = nmf_model.fit_transform(user_item)
        H = nmf_model.components_
        predicted_ratings = np.dot(W, H)
        use_collab = True
        app.logger.info("Collaborative filtering enabled")
    except FileNotFoundError:
        pop_scores = df_clean['ratings_count_clean'].values
        use_collab = False
        app.logger.info("Using popularity-based recommendations")

    # TF-IDF content model
    vectorizer = TfidfVectorizer(
        stop_words='english', max_features=5000,
        ngram_range=(1,2), min_df=2, max_df=0.8
    )
    tfidf_matrix = vectorizer.fit_transform(df_clean['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Title-to-index mapping
    indices = pd.Series(df_clean.index, index=df_clean['title']).drop_duplicates()

    return True

def normalize(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx - mn == 0:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn + 1e-9)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json(silent=True) or {}
        term = data.get('query', '')
        if not isinstance(term, str) or not term.strip():
            return jsonify({'results': []})
        term_lower = term.strip().lower()

        mask_title = df_clean['title'].str.lower().str.contains(term_lower, na=False, regex=False)
        mask_author = df_clean['authors_clean'].str.lower().str.contains(term_lower, na=False, regex=False)
        mask_cat = df_clean['categories_clean'].str.lower().str.contains(term_lower, na=False, regex=False)
        matches = df_clean[mask_title | mask_author | mask_cat].head(10)

        results = []
        for _, row in matches.iterrows():
            results.append({
                'title': row['title'],
                'authors_clean': row['authors_clean'],
                'categories_clean': row['categories_clean'],
                'average_rating_clean': float(row.get('average_rating_clean', 0))
            })
        return jsonify({'results': results})
    except Exception as e:
        app.logger.error(f"Search error: {e}", exc_info=True)
        return jsonify({'error': 'Search failed'}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json(silent=True) or {}
        title = data.get('title', '').strip()
        user_id = data.get('user_id', 1)
        K = data.get('K', 10)
        w_content = data.get('w_content', 0.6)
        w_collab = data.get('w_collab', 0.4)

        if title not in indices:
            return jsonify({'error': f"Book '{title}' not found"}), 404
        idx = indices[title]

        c_scores = cosine_sim[idx]
        c_norm = normalize(c_scores)

        if use_collab and predicted_ratings is not None:
            if not (1 <= user_id <= predicted_ratings.shape[0]):
                return jsonify({'error': f"User ID {user_id} out of range"}), 400
            collab = predicted_ratings[user_id - 1]
        else:
            collab = pop_scores
        p_norm = normalize(collab)

        total_w = w_content + w_collab
        if total_w > 0:
            w_content /= total_w
            w_collab /= total_w

        hybrid = w_content * c_norm + w_collab * p_norm
        hybrid[idx] = -np.inf
        top_idx = np.argsort(hybrid)[::-1][:K]

        recs = []
        for i in top_idx:
            row = df_clean.iloc[i]
            recs.append({
                'title': row['title'],
                'authors_clean': row['authors_clean'],
                'categories_clean': row['categories_clean'],
                'similarity_score': round(float(hybrid[i]), 3),
                'average_rating_clean': float(row.get('average_rating_clean', 0))
            })
        return jsonify({'recommendations': recs})
    except Exception as e:
        app.logger.error(f"Recommendation error: {e}", exc_info=True)
        return jsonify({'error': 'Recommendation failed'}), 500

@app.route('/random')
def random_books():
    try:
        if df_clean is None or df_clean.empty:
            return jsonify({'books': []})
        sample = df_clean.sample(min(12, len(df_clean)))
        books = []
        for _, row in sample.iterrows():
            books.append({
                'title': row['title'],
                'authors_clean': row['authors_clean'],
                'categories_clean': row['categories_clean'],
                'average_rating_clean': float(row.get('average_rating_clean', 0))
            })
        return jsonify({'books': books})
    except Exception as e:
        app.logger.error(f"Random books error: {e}", exc_info=True)
        return jsonify({'books': []})

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok' if df_clean is not None else 'error',
        'books_loaded': len(df_clean) if df_clean is not None else 0,
        'collab_mode': use_collab,
        'csv_exists': os.path.exists(CSV_PATH),
        'csv_path': CSV_PATH
    })

if __name__ == '__main__':
    if load_data():
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
    else:
        app.logger.error("Failed to initialize data. Exiting.")
