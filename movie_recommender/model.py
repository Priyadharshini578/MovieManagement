# model.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

class RecommenderSystem:
    def __init__(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings
        self.movie_features = None
        self.svd = None
        self.utility_matrix = None
        self._prepare()

    def _prepare(self):
        # Content-based filtering: Genres to dummy matrix
        genre_matrix = self.movies['genres'].str.get_dummies('|')
        self.movie_features = genre_matrix

        # Collaborative Filtering: Create utility matrix
        self.utility_matrix = self.ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

        # SVD model
        self.svd = TruncatedSVD(n_components=20, random_state=42)
        self.svd_matrix = self.svd.fit_transform(self.utility_matrix)

    def get_content_based_recommendations(self, movie_title, top_n=5):
        if movie_title not in self.movies['title'].values:
            return []

        idx = self.movies[self.movies['title'] == movie_title].index[0]
        cosine_sim = cosine_similarity(
            [self.movie_features.iloc[idx]], self.movie_features
        )[0]
        similar_indices = cosine_sim.argsort()[::-1][1:top_n + 1]
        return self.movies.iloc[similar_indices][['movieId', 'title']]

    def get_collaborative_recommendations(self, user_id, top_n=5):
        if user_id not in self.utility_matrix.index:
            return []

        user_ratings = self.utility_matrix.loc[user_id].values.reshape(1, -1)
        user_proj = self.svd.transform(user_ratings)
        sim_scores = cosine_similarity(user_proj, self.svd_matrix)[0]

        similar_users = sim_scores.argsort()[::-1][1:top_n + 1]
        similar_users_ratings = self.utility_matrix.iloc[similar_users]
        mean_ratings = similar_users_ratings.mean().sort_values(ascending=False)

        recommended = mean_ratings.head(top_n).index
        return self.movies[self.movies['movieId'].isin(recommended)][['movieId', 'title']]
