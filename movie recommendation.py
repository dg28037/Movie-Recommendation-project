import pandas as pd
import re
df = pd.read_csv("movies.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
print(df.duplicated().sum())
print(df["movieId"].nunique())
print(df["title"].nunique())
print(df["genres"].nunique())
print(df["genres"].head(10))
print(df["genres"].isnull().sum())

def split_genres(text):
    text = text.split("|")
    print(text)
    return text
df["seperator"] = df["genres"].apply(split_genres)
print(df[["genres" , "seperator"]].head())

def create_final_text(text):
    return " ".join(text)
df["final_sentence"] = df["seperator"].apply(create_final_text)
print(df[["seperator" , "final_sentence"]].head())

from sklearn.feature_extraction.text import TfidfVectorizer
model = TfidfVectorizer()
X = model.fit_transform(df["final_sentence"])
print(X.shape)
print(model.get_feature_names_out()[:20])
print(X.toarray())
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(X)
print(similarity)

movie_name = input("Enter movie name: ")
movie_index = df[df["title"] == movie_name].index[0]
similar_movies = similarity[movie_index].argsort()[::-1][1:6]
print("\nRecommended Movies:")
for index in similar_movies:
    print(df.iloc[index]["title"])
