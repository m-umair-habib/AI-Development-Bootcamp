# ----- Developed By: Muhammad Umair Habib -----

# Import necessary libraries
import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.corpus import stopwords
import random

# Download the NLTK data files
nltk.download('movie_reviews')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Preprocess the dataset and extract the features
def extract_features(words):
    return {word: True for word in words}

# Load the movie_reviews dataset from NLTK
documents = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                    for fileid in movie_reviews.fileids(category)]

# Shuffle the dataset to ensure random distribution
random.shuffle(documents)

# Prepare the dataset for training and testing
featuresets = [(extract_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[: 1600], featuresets[1600:]

# Train the Nive Bayes Classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate the Classifier on the test set
accuracy = nltk_accuracy(classifier, test_set)
print(f"Accuracy: {accuracy * 100: 2f}%")

# Show the most informative features
classifier.show_most_informative_features(10)

# Test on new input sentences
def analyze_sentiment(text):
    # Tokenize and remove stopwords
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words('english')]

    # Predict the sentiment
    features = extract_features(words)
    return classifier.classify(features)

# Test the classifier with some custom test inputs
test_sentences = [
    "This movie is absolutely fantastic! The acting, the story, everything was amazing!",
    "I hated this movie. It was the waste of time and money.",
    "The plot was a bit dull, but the performance were great.",
    "I have mixed feelings about this film. It was okay, not great but not terrible either."
]

for sentence in test_sentences:
    print(f"Sentence: {sentence}")
    print(f"Predicted Sentiment: {analyze_sentiment(sentence)}")
    print()