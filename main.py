from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np

class Query(BaseModel):
    query: str

app = FastAPI()

# Load the trained TensorFlow model, vectorizer, and label encoder
model = tf.keras.models.load_model('models/tf_intent_recognition_model.keras')  # Use .keras extension as per the save method
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
label_encoder = joblib.load('models/label_encoder.joblib')

def score_response(response: str) -> int:
    score = 0
    keywords = ["Python", "Java", "JavaScript", "experience", "project", "cloud"]
    word_count = {}
    words = response.lower().split()
    
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    
    for keyword in keywords:
        if keyword.lower() in word_count:
            score += min(word_count[keyword.lower()], 2) * 10
            if word_count[keyword.lower()] > 2:
                score -= (word_count[keyword.lower()] - 2) * 5

    score += len(words) / 2
    return max(min(score, 100), 0)

@app.post("/predict")
def predict_intent(query: Query):
    try:
        # Preprocess and vectorize the input query
        X = vectorizer.transform([query.query])
        # Predict the intent
        predictions = model.predict(X.toarray())
        predicted_intent = label_encoder.inverse_transform([np.argmax(predictions)])[0]
        probabilities = {label_encoder.classes_[i]: float(predictions[0][i]) for i in range(len(label_encoder.classes_))}
        # Score the response
        score = score_response(query.query)
        return {"intent": predicted_intent, "score": score, "probabilities": probabilities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
