import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import joblib
import os

# Load the preprocessed data
with open('data/processed_data.json', 'r') as f:
    data = json.load(f)

# Extract text and labels
texts = [item['text'] for item in data['responses']]
intents = [item['intent'] for item in data['responses']]

# Encode labels as integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(intents)

# Initialize the TF-IDF vectorizer and transform the texts
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model, vectorizer, and label encoder
os.makedirs('models', exist_ok=True)
model.save('models/tf_intent_recognition_model.keras')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
joblib.dump(label_encoder, 'models/label_encoder.joblib')
