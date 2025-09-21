import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static', template_folder='templates')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'static', 'data', 'books_clean.csv')
RATINGS_PATH = os.path.join(BASE_DIR, 'static', 'data', 'ratings.csv')

df_clean = None
tfidf_matrix = None
cosine_sim = None
indices = None
use_collab = False
pop_scores = None
predicted_ratings = None

def load_data():
    global df_clean, tfidf_matrix, cosine_sim, indices, use_collab, pop_scores, predicted_ratings

    print(f"Loading CSV from: {CSV_PATH}")
    print(f"CSV exists: {os.path.exists(CSV_PATH)}")

    try:
        df_clean = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(df_clean)} books")
    except Exception as e:
        print(f"Failed to load books_clean.csv: {e}")
        df_clean = pd.DataFrame(columns=[
            'title', 'authors_clean', 'categories_clean',
            'combined_features', 'description',
            'average_rating_clean', 'ratings_count_clean'
        ])

    for col in ['title', 'authors_clean', 'categories_clean', 'combined_features', 'description']:
        df_clean[col] = df_clean.get(col, '').fillna('').astype(str)

    df_clean['average_rating_clean'] = df_clean.get('average_rating_clean', 0.0).fillna(0.0).astype(float)
    df_clean['ratings_count_clean'] = df_clean.get('ratings_count_clean', 0).fillna(0).astype(int)

    try:
        ratings_df = pd.read_csv(RATINGS_PATH)
        user_item = ratings_df.pivot_table(
            index='user_id', columns='book_id', values='rating'
        ).fillna(0)
        nmf_model = NMF(n_components=40, random_state=50)
        W = nmf_model.fit_transform(user_item)
        H = nmf_model.components_
        predicted_ratings = np.dot(W, H)
        use_collab = True
        print("Collaborative filtering model loaded")
    except Exception:
        pop_scores = df_clean['ratings_count_clean'].values
        use_collab = False
        print("Using popularity-based recommendations")

    vectorizer = TfidfVectorizer(
        stop_words='english', max_features=10000,
        ngram_range=(2, 3), min_df=3, max_df=0.9
    )
    tfidf_matrix = vectorizer.fit_transform(df_clean['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    print("TF-IDF model created")

    indices = pd.Series(df_clean.index, index=df_clean['title']).drop_duplicates()
    print("Title mapping created")

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
        mask_auth = df_clean['authors_clean'].str.lower().str.contains(term_lower, na=False, regex=False)
        mask_cat = df_clean['categories_clean'].str.lower().str.contains(term_lower, na=False, regex=False)
        matches = df_clean[mask_title | mask_auth | mask_cat].head(10)

        results = [{
            'title': row['title'],
            'authors_clean': row['authors_clean'],
            'categories_clean': row['categories_clean'],
            'average_rating_clean': float(row['average_rating_clean'])
        } for _, row in matches.iterrows()]

        return jsonify({'results': results})
    except Exception as e:
        app.logger.error(f"Search error: {e}", exc_info=True)
        return jsonify({'error': 'Search failed'}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json(silent=True) or {}
        title = data.get('title', '').strip()
        user_id = int(data.get('user_id', 1))
        K = int(data.get('K', 15))
        w_content = float(data.get('w_content', 0.8))
        w_collab = float(data.get('w_collab', 0.2))

        if title not in indices:
            return jsonify({'error': f"Book '{title}' not found"}), 404
        idx = indices[title]

        content_scores = cosine_sim[idx]
        c_norm = normalize(content_scores)

        if use_collab and predicted_ratings is not None:
            if not (1 <= user_id <= predicted_ratings.shape[0]):
                return jsonify({'error': f"User ID {user_id} out of range"}), 400
            collab_scores = predicted_ratings[user_id - 1]
        else:
            collab_scores = pop_scores if pop_scores is not None else np.ones(len(df_clean))
        p_norm = normalize(collab_scores)

        total_w = w_content + w_collab
        if total_w > 0:
            w_content /= total_w
            w_collab /= total_w

        hybrid = w_content * c_norm + w_collab * p_norm
        hybrid[idx] = -np.inf

        top_idx = np.argsort(hybrid)[::-1][:K]
        recommendations = [{
            'title': df_clean.iloc[i]['title'],
            'authors_clean': df_clean.iloc[i]['authors_clean'],
            'categories_clean': df_clean.iloc[i]['categories_clean'],
            'similarity_score': round(float(hybrid[i]), 3),
            'average_rating_clean': float(df_clean.iloc[i]['average_rating_clean'])
        } for i in top_idx if hybrid[i] > -np.inf]

        return jsonify({'recommendations': recommendations})
    except Exception as e:
        app.logger.error(f"Recommendation error: {e}", exc_info=True)
        return jsonify({'error': 'Recommendation failed'}), 500

@app.route('/random')
def random_books():
    try:
        if df_clean is None or df_clean.empty:
            return jsonify({'books': []})
        sample = df_clean.sample(min(12, len(df_clean)))
        books = [{
            'title': row['title'],
            'authors_clean': row['authors_clean'],
            'categories_clean': row['categories_clean'],
            'average_rating_clean': float(row['average_rating_clean'])
        } for _, row in sample.iterrows()]
        return jsonify({'books': books})
    except Exception as e:
        app.logger.error(f"Random books error: {e}", exc_info=True)
        return jsonify({'books': []}), 500

@app.route('/all-books')
def all_books():
    if df_clean is None or df_clean.empty:
        return jsonify({'error': 'No books available'}), 500
    sorted_df = df_clean.sort_values('title')
    books = []
    for _, row in sorted_df.iterrows():
        desc = row.get('description', '')
        snippet = desc[:150].rstrip()
        if len(desc) > 150:
            snippet += '…'
        books.append({
            'title': row['title'],
            'author': row['authors_clean'],
            'snippet': snippet
        })
    return jsonify({'books': books})

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy' if df_clean is not None else 'error',
        'books_loaded': len(df_clean) if df_clean is not None else 0,
        'collab_mode': use_collab,
        'tfidf_ready': tfidf_matrix is not None
    })

if __name__ == '__main__':
    load_data()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
