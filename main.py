from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import json

class Query(BaseModel):
    query: str

app = FastAPI()

# Load the trained model and vectorizer
model = joblib.load('models/intent_recognition_model.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

def score_response(response: str) -> int:
    # Simple scoring based on length and presence of specific keywords
    score = 0
    keywords = ["Python", "Java", "JavaScript", "experience", "project", "cloud"]
    for keyword in keywords:
        if keyword.lower() in response.lower():
            score += 10
    # Additional points for response length (demonstrating thoroughness)
    score += len(response.split()) / 2
    return min(score, 100)  # Limit score to a maximum of 100


# Function to save candidate data
def save_candidate_data(candidate_data: dict):
    os.makedirs('data', exist_ok=True)
    with open('data/candidate_data.json', 'a') as f:
        json.dump(candidate_data, f)
        f.write('\n')


@app.post("/predict")
def predict_intent(query: Query):
    try:
        # Preprocess and vectorize the input query
        X = vectorizer.transform([query.query])
        # Predict the intent
        intent = model.predict(X)[0]
        # Score the response
        score = score_response(query.query)
        # Save candidate data
        candidate_data = {"query": query.query, "intent": intent, "score": score}
        save_candidate_data(candidate_data)
        return {"intent": intent, "score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))