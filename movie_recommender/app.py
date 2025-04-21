# app.py
import streamlit as st
from utils import load_data
from model import RecommenderSystem

st.title("🎬 Movie Recommendation System")
st.write("Hybrid system using Content-based and Collaborative Filtering (SVD)")

# Load data
movies, ratings = load_data()
recommender = RecommenderSystem(movies, ratings)

option = st.radio("Choose Recommendation Type:", ('Content-Based', 'Collaborative Filtering'))

if option == 'Content-Based':
    selected_movie = st.selectbox("Pick a movie:", movies['title'].values)
    if st.button("Recommend"):
        results = recommender.get_content_based_recommendations(selected_movie)
        st.write("### Recommended Movies:")
        for title in results['title']:
            st.write(f"- {title}")

else:
    user_ids = ratings['userId'].unique()
    selected_user = st.selectbox("Select User ID:", user_ids)
    if st.button("Recommend"):
        results = recommender.get_collaborative_recommendations(selected_user)
        st.write("### Recommended Movies:")
        for title in results['title']:
            st.write(f"- {title}")
