import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up page title
st.set_page_config(page_title="Movie Recommendation System", layout="centered")
st.title("🎬 Movie Recommendation System")
st.write("Get instant movie recommendations based on genre similarity.")


# Load and cache data to prevent reloading on every click
@st.cache_data
def load_data():
    # Update "movies.csv" with your actual file path if needed
    df = pd.read_csv("movies.csv")

    # Clean and process genres
    df["genres"] = df["genres"].fillna("")
    df["seperator"] = df["genres"].apply(lambda x: x.split("|"))
    df["final_sentence"] = df["seperator"].apply(lambda x: " ".join(x))
    return df


@st.cache_data
def calculate_similarity(sentences):
    model = TfidfVectorizer()
    X = model.fit_transform(sentences)
    similarity = cosine_similarity(X)
    return similarity


# Initialize data and models
try:
    df = load_data()
    similarity = calculate_similarity(df["final_sentence"])

    # Sidebar overview stats (Optional)
    st.sidebar.header("Dataset Overview")
    st.sidebar.metric("Total Movies", f"{df.shape[0]:,}")
    st.sidebar.metric("Unique Genres", f'{df["genres"].nunique():,}')

    # Dropdown menu for selecting a movie (prevents typing errors)
    movie_list = df["title"].values
    selected_movie = st.selectbox(
        "Type or select a movie you like:", movie_list
    )

    # Recommendation logic triggered by button
    if st.button("Recommend"):
        # Get index of the selected movie
        movie_index = df[df["title"] == selected_movie].index[0]

        # Get top 5 similar movies (excluding the movie itself)
        similar_movies_indices = similarity[movie_index].argsort()[::-1][1:6]

        st.subheader("Recommended Movies:")

        # Display recommendations cleanly
        for idx, index in enumerate(similar_movies_indices, start=1):
            recommended_title = df.iloc[index]["title"]
            recommended_genres = df.iloc[index]["genres"].replace("|", ", ")
            st.markdown(f"**{idx}. {recommended_title}**")
            st.caption(f"Genres: {recommended_genres}")

except FileNotFoundError:
    st.error(
        "Error: 'movies.csv' file not found. Please make sure the file is in the same directory as this script."
    )
