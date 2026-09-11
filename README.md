# MOVIE-RECOMMENDATION-SYSTEM
An end-to-end machine learning project that builds a metadata-driven recommendation engine using Natural Language Processing (NLP). The system creates a unique textual profile for over 4,800 films and utilizes vector space modeling to instantly recommend the top 5 closest matching movies based on user preferences.

## How the Pipeline Works
1. **Feature Merging:** Extracts and cleans features from the TMDB 5000 dataset (plot, genres, keywords, cast, and crew) into a unified `tags` feature.
2. **Token Normalization:** Space-strips entity names (e.g., converting "Johnny Depp" to `johnnydepp`) to prevent word collisions.
3. **Text Stemming:** Applies NLTK's `PorterStemmer` to reduce words to their root forms (e.g., "loving", "liked" $\rightarrow$ `love`).
4. **Vector Quantization:** Converts textual tags into numerical coordinates using `CountVectorizer`, generating a 5,000-dimensional sparse array.
5. **Similarity Evaluation:** Computes spatial proximity using **Cosine Similarity** to filter and drop the top 5 closest matching film IDs upon query.

## Tech Stack & Libraries
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **NLP & Text Processing:** NLTK (PorterStemmer)
* **Machine Learning:** Scikit-Learn (CountVectorizer, Cosine Similarity)
* **Environment:** Jupyter Notebook / Anaconda

## Key Business Takeaways
* **Cold-Start Resolution:** Recommends brand-new or zero-interaction movies instantly without requiring user ratings.
* **Catalog Optimization:** Maximizes long-tail catalog utilization by driving recommendations for niche, highly relevant content.
