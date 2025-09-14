class BookRecommender {
    constructor() {
        this.searchInput = document.getElementById('searchInput');
        this.searchBtn = document.getElementById('searchBtn');
        this.searchResults = document.getElementById('searchResults');
        this.recommendationsContainer = document.getElementById('recommendationsContainer');
        this.randomBtn = document.getElementById('randomBtn');
        this.randomBooks = document.getElementById('randomBooks');
        
        this.initializeEventListeners();
        this.loadRandomBooks();
    }
    
    initializeEventListeners() {
        this.searchBtn.addEventListener('click', () => this.searchBooks());
        this.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.searchBooks();
        });
        this.randomBtn.addEventListener('click', () => this.loadRandomBooks());
    }
    
    async searchBooks() {
        const query = this.searchInput.value.trim();
        if (!query) return;
        
        this.showLoading(this.searchResults);
        
        try {
            const response = await fetch('/search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query})
            });
            
            const data = await response.json();
            
            if (data.error) {
                this.showError(this.searchResults, data.error);
                return;
            }
            
            this.displaySearchResults(data.results);
            
        } catch (error) {
            console.error('Search error:', error);
            this.showError(this.searchResults, 'Failed to search books. Please try again.');
        }
    }
    
    async getRecommendations(bookTitle) {
        this.showLoading(this.recommendationsContainer);
        
        try {
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    title: bookTitle,
                    user_id: 1,
                    K: 8,
                    w_content: 0.7,
                    w_collab: 0.3
                })
            });
            
            const data = await response.json();
            
            if (data.error) {
                this.showError(this.recommendationsContainer, data.error);
                return;
            }
            
            this.displayRecommendations(bookTitle, data.recommendations);
            
        } catch (error) {
            console.error('Recommendation error:', error);
            this.showError(this.recommendationsContainer, 'Failed to get recommendations. Please try again.');
        }
    }
    
    async loadRandomBooks() {
        this.showLoading(this.randomBooks);
        
        try {
            const response = await fetch('/random');
            const data = await response.json();
            this.displayRandomBooks(data.books);
        } catch (error) {
            console.error('Random books error:', error);
            this.showError(this.randomBooks, 'Failed to load random books.');
        }
    }
    
    displaySearchResults(books) {
        if (!books || books.length === 0) {
            this.searchResults.innerHTML = '<p class="error">No books found. Try different keywords.</p>';
            return;
        }
        
        const html = books.map(book => `
            <div class="book-card" onclick="recommender.getRecommendations('${book.title.replace(/'/g, "\\'")}')">
                <div class="book-title">${book.title}</div>
                <div class="book-author">by ${book.authors_clean}</div>
                <div class="book-category">📚 ${book.categories_clean}</div>
                <div class="book-rating">⭐ ${book.average_rating_clean}</div>
            </div>
        `).join('');
        
        this.searchResults.innerHTML = `
            <h3>Search Results (click on a book to get recommendations):</h3>
            ${html}
        `;
    }
    
    displayRecommendations(originalTitle, recommendations) {
        if (!recommendations || recommendations.length === 0) {
            this.recommendationsContainer.innerHTML = '<p class="error">No recommendations found.</p>';
            return;
        }
        
        const html = recommendations.map(book => `
            <div class="book-card">
                <div class="book-title">
                    ${book.title}
                    <span class="similarity-score">${Math.round(book.similarity_score * 100)}% match</span>
                </div>
                <div class="book-author">by ${book.authors_clean}</div>
                <div class="book-category">📚 ${book.categories_clean}</div>
                <div class="book-rating">⭐ ${book.average_rating_clean}</div>
            </div>
        `).join('');
        
        this.recommendationsContainer.innerHTML = `
            <h2>🎯 Recommendations for "${originalTitle}"</h2>
            ${html}
        `;
        
        this.recommendationsContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    displayRandomBooks(books) {
        if (!books || books.length === 0) {
            this.randomBooks.innerHTML = '<p class="error">Failed to load random books.</p>';
            return;
        }
        
        const html = books.map(book => `
            <div class="book-card" onclick="recommender.getRecommendations('${book.title.replace(/'/g, "\\'")}')">
                <div class="book-title">${book.title}</div>
                <div class="book-author">by ${book.authors_clean}</div>
                <div class="book-category">📚 ${book.categories_clean}</div>
                <div class="book-rating">⭐ ${book.average_rating_clean}</div>
            </div>
        `).join('');
        
        this.randomBooks.innerHTML = html;
    }
    
    showLoading(element) {
        element.innerHTML = '<div class="loading">🔄 Loading...</div>';
    }
    
    showError(element, message) {
        element.innerHTML = `<div class="error">${message}</div>`;
    }
}

// Initialize the app when the page loads
let recommender;
document.addEventListener('DOMContentLoaded', () => {
    recommender = new BookRecommender();
});
