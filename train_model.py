import pandas as pd
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("bbc-text.csv")

# Stopwords
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):

    text = text.lower()

    tokens = word_tokenize(text)

    filtered_tokens = []

    for word in tokens:

        if (
            word not in stop_words
            and word not in string.punctuation
            and word.isalpha()
        ):
            filtered_tokens.append(word)

    return " ".join(filtered_tokens)

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Features and Labels
X = df['cleaned_text']
y = df['category']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)

X_vectorized = vectorizer.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = MultinomialNB()

# Train
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# Save model
pickle.dump(model, open("news_model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n✅ Files saved successfully!")