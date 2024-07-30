import json
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Load the preprocessed data
with open('data/processed_data.json', 'r') as f:
    data = json.load(f)

# Extract text and labels
texts = [item['text'] for item in data['responses']]
intents = [item['intent'] for item in data['responses']]

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the texts to feature vectors
X = vectorizer.fit_transform(texts)
y = intents

# Verify the results
print("Feature matrix shape:", X.shape)
print("Sample feature names:", vectorizer.get_feature_names_out()[:10])

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Save the vectorizer for future use
joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
