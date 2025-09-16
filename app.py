from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables
df_clean = None
tfidf_matrix = None
cosine_sim = None
indices = None
use_collab = False
pop_scores = None
predicted_ratings = None

def load_data():
    global df_clean, tfidf_matrix, cosine_sim, indices, use_collab, pop_scores, predicted_ratings
    
    try:
        print("🔍 Current working directory:", os.getcwd())
        print("🔍 Files in current directory:", os.listdir('.'))
        
        csv_path = os.path.join(os.path.dirname(__file__), 'static', 'data', 'books_clean.csv')
        print(f"🔍 Looking for CSV at: {csv_path}")
        print(f"🔍 CSV file exists: {os.path.exists(csv_path)}")
        
        df_clean = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df_clean)} books")
        
        # Ensure ratings columns exist
        if 'average_rating_clean' not in df_clean.columns:
            df_clean['average_rating_clean'] = 0.0
        if 'ratings_count_clean' not in df_clean.columns:
            df_clean['ratings_count_clean'] = 0
            
        print(f"Loaded {len(df_clean)} books")
        
        # Try collaborative filtering
        try:
            ratings_df = pd.read_csv('static/data/ratings.csv')
            user_item = ratings_df.pivot_table(
                index='user_id', columns='book_id', values='rating'
            ).fillna(0)
            nmf_model = NMF(n_components=20, random_state=42)
            W = nmf_model.fit_transform(user_item)
            H = nmf_model.components_
            predicted_ratings = np.dot(W, H)
            use_collab = True
            print("Collaborative filtering model loaded")
        except FileNotFoundError:
            pop_scores = df_clean['ratings_count_clean'].values
            use_collab = False
            print("Using popularity-based recommendations")
        
        # Build TF-IDF model
        print("Building TF-IDF model...")
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(2,3),
            min_df=5,
            max_df=0.9
        )
        tfidf_matrix = vectorizer.fit_transform(df_clean['combined_features'])
        cosine_sim = cosine_similarity(tfidf_matrix)
        print("TF-IDF model created")
        
        # Create title mapping
        indices = pd.Series(df_clean.index, index=df_clean['title']).drop_duplicates()
        print("Title mapping created")
        
        return True
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0,1]"""
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
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        term = data.get('query', '').strip()
        if not term:
            return jsonify({'results': []})
        
        term_lower = term.lower()
        print(f"Search term: '{term_lower}'")
        
        # Search in multiple columns
        title_mask = df_clean['title'].str.lower().str.contains(term_lower, na=False, regex=False)
        author_mask = df_clean['authors_clean'].str.lower().str.contains(term_lower, na=False, regex=False)
        category_mask = df_clean['categories_clean'].str.lower().str.contains(term_lower, na=False, regex=False)
        
        matches = df_clean[title_mask | author_mask | category_mask].head(10)
        
        print(f"Found {len(matches)} matches")
        
        # Format results
        results = []
        for _, row in matches.iterrows():
            results.append({
                'title': str(row['title']),
                'authors_clean': str(row['authors_clean']),
                'categories_clean': str(row['categories_clean']),
                'average_rating_clean': float(row.get('average_rating_clean', 0))
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        title = data.get('title', '').strip()
        user_id = data.get('user_id', 1)
        K = data.get('K', 8)
        w_content = data.get('w_content', 0.7)
        w_collab = data.get('w_collab', 0.3)
        
        print(f"Getting recommendations for: '{title}'")
        
        if title not in indices:
            return jsonify({'error': f"Book '{title}' not found"}), 404
        
        idx = indices[title]
        
        # Content-based scores
        content_scores = cosine_sim[idx]
        c_norm = normalize(content_scores)
        
        # Collaborative or popularity scores
        if use_collab and predicted_ratings is not None:
            if not (0 <= user_id - 1 < predicted_ratings.shape[0]):
                return jsonify({'error': f"User ID {user_id} out of range"}), 400
            collab_scores = predicted_ratings[user_id - 1]
        else:
            collab_scores = pop_scores if pop_scores is not None else np.ones(len(df_clean))
        
        p_norm = normalize(collab_scores)
        
        # Hybrid scoring
        total_w = w_content + w_collab
        if total_w > 0:
            w_content /= total_w
            w_collab /= total_w
        
        hybrid = w_content * c_norm + w_collab * p_norm
        hybrid[idx] = -np.inf  # Exclude original book
        
        # Get top recommendations
        top_idx = np.argsort(hybrid)[::-1][:K]
        
        recommendations = []
        for i in top_idx:
            if hybrid[i] > -np.inf:  # Valid recommendation
                row = df_clean.iloc[i]
                recommendations.append({
                    'title': str(row['title']),
                    'authors_clean': str(row['authors_clean']),
                    'categories_clean': str(row['categories_clean']),
                    'similarity_score': round(float(hybrid[i]), 3),
                    'average_rating_clean': float(row.get('average_rating_clean', 0))
                })
        
        print(f"Generated {len(recommendations)} recommendations")
        return jsonify({'recommendations': recommendations})
        
    except Exception as e:
        print(f"Recommendation error: {e}")
        return jsonify({'error': 'Recommendation failed'}), 500

@app.route('/random')
def random_books():
    try:
        if df_clean is None or len(df_clean) == 0:
            return jsonify({'books': []})
        
        sample_size = min(12, len(df_clean))
        sample = df_clean.sample(sample_size)
        
        books = []
        for _, row in sample.iterrows():
            books.append({
                'title': str(row['title']),
                'authors_clean': str(row['authors_clean']),
                'categories_clean': str(row['categories_clean']),
                'average_rating_clean': float(row.get('average_rating_clean', 0))
            })
        
        print(f"Generated {len(books)} random books")
        return jsonify({'books': books})
        
    except Exception as e:
        print(f"Random books error: {e}")
        return jsonify({'books': []})

@app.route('/health')
def health():
    csv_path = os.path.join(os.path.dirname(__file__), 'static', 'data', 'books_clean.csv')
    status = {
        'status': 'healthy' if df_clean is not None else 'error',
        'books_loaded': len(df_clean) if df_clean is not None else 0,
        'collab_mode': use_collab,
        'tfidf_ready': tfidf_matrix is not None,
        'csv_exists': os.path.exists(csv_path),
        'current_dir': os.getcwd(),
        'csv_path': csv_path
    }
    return jsonify(status)


if __name__ == '__main__':
    print("Starting Book Recommender App...")
    if load_data():
        print("Server ready!")
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)  # debug=False for production
    else:
        print("Failed to initialize. Check your data files.")
