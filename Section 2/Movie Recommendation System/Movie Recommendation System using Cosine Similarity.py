# ----- Developed By: Muhammad Umair Habib -----

# Import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Dataset
data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['The Matrix', 'John Wick', 'The Godfather', 'Pulp Fiction', 'The Dark Knight'],
    'genre': ['Action, Sci-Fi', 'Action, Thriller', 'Crime, Drama', 'Crime, Drama', 'Action, Crime, Drama']
}

# Convert dataset into dataframe
df = pd.DataFrame(data)

# Display the dataset
print("Movie Data: ")
print(df)

# Define a TF-TDF Vectorizer to transform the genre text into vectors
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the genre column into matrix of TF-IDF features
tfidf_matrix = tfidf.fit_transform(df['genre'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on cosine similarity
def get_recommendations(title, cosine_sim = cosine_sim):
    # Get the index of the movie that matches the title
    idx = df[df['title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indexes of the two most similar movies
    sim_scores = sim_scores[1:3]

    # Get the movies indexes
    movies_indices = [i[0] for i in sim_scores]

    # Return titles of the most similar movies
    return df['title'].iloc[movies_indices]

# Test the recommendation system with an example
movie_title = 'The Matrix'
recommendation_movies = get_recommendations(movie_title)

print(f"Movie recommendation for '{movie_title}': ")

for movie in recommendation_movies:
    print(movie)