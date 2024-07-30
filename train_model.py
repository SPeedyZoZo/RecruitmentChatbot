import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Load the preprocessed data
with open('data/processed_data.json', 'r') as f:
    data = json.load(f)

# Extract text and labels
texts = [item['text'] for item in data['responses']]
intents = [item['intent'] for item in data['responses']]

# Check class distribution
intent_counts = Counter(intents)
print(f"Intent counts: {intent_counts}")

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the texts to feature vectors
X = vectorizer.fit_transform(texts)
y = intents

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the trained model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/intent_recognition_model.joblib')
