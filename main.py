import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data
data = pd.read_csv('kdrama.csv')

# Data preprocessing
for col in ['Description', 'Genre', 'Tags', 'Actors', 'Title', 'Year of release', 'Number of Episodes', 'Rating', 'Rank']:
    data[col] = data[col].fillna('')

# Ensure numeric columns are properly formatted
data['Year of release'] = pd.to_numeric(data['Year of release'], errors='coerce')
data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')

# Combine features
data['combined_features'] = data['Description'] + " " + data['Genre'] + " " + data['Tags'] + " " + data['Actors']

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(data.index, index=data['Title']).drop_duplicates()

# Function for recommendation
def recommend(item_name, feature_type='Title'):
    if feature_type == 'Title':
        idx = indices.get(item_name)
        if idx is None:
            return pd.DataFrame()
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return data.iloc[movie_indices]
    else:
        # Filter dramas that contain the input in selected feature
        filtered = data[data[feature_type].str.contains(item_name, case=False, na=False)]
        return filtered.head(10)  # show top 10

# ---- Streamlit UI ----
st.title('K-Drama Recommendation System')

option = st.selectbox('Recommend based on:', ['Title', 'Genre', 'Tags', 'Actors'])

# Dynamic autocomplete input
if option == 'Title':
    query = st.text_input('Enter Drama Title')
    suggestions = [title for title in data['Title'] if query.lower() in title.lower()]
elif option == 'Genre':
    query = st.text_input('Enter Genre')
    genre_list = list(set(genre.strip() for sublist in data['Genre'].str.split(',') for genre in sublist if genre))
    suggestions = [g for g in genre_list if query.lower() in g.lower()]
elif option == 'Tags':
    query = st.text_input('Enter Tag')
    tag_list = list(set(tag.strip() for sublist in data['Tags'].str.split(',') for tag in sublist if tag))
    suggestions = [t for t in tag_list if query.lower() in t.lower()]
else:
    query = st.text_input('Enter Actor Name')
    actor_list = list(set(actor.strip() for sublist in data['Actors'].str.split(',') for actor in sublist if actor))
    suggestions = [a for a in actor_list if query.lower() in a.lower()]

# Auto-complete dropdown
if query:
    selected = st.selectbox(f"Select {option}", suggestions)

    # Filters
    rating_filter = st.slider('Select Minimum Rating', 1, 10, 7)

    # Year range filter
    min_year, max_year = int(data['Year of release'].min()), int(data['Year of release'].max())
    selected_year_range = st.slider('Select Year Range', min_year, max_year, (min_year, max_year))

    if selected:
        st.write(f"Recommendations based on {option}: **{selected}**")

        recommended = recommend(selected, feature_type=option)
        filtered_recommendations = recommended[
            (recommended['Rating'] >= rating_filter) &
            (recommended['Year of release'].between(selected_year_range[0], selected_year_range[1]))
        ]

        for idx, movie in filtered_recommendations.iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(movie['Title'])
                    st.write(f"**Rating**: {movie['Rating']}")
                    st.write(f"**Year of Release**: {movie['Year of release']}")
                    st.write(f"**Number of Episodes**: {movie['Number of Episodes']}")
                    st.write(f"**Description**: {movie['Description'][:300]}...")
                    st.markdown("---")
