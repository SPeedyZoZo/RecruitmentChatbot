from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

class Query(BaseModel):
    query: str

app = FastAPI()

# Load the trained model and vectorizer
model = joblib.load('models/intent_recognition_model.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')


@app.post("/predict")
def predict_intent(query: Query):
    try:
        # Preprocess and vectorize the input query
        X = vectorizer.transform([query.query])
        # Debug: Print transformed input features
        print(f"Transformed input features: {X.toarray()}")
        # Predict the intent
        intent = model.predict(X)[0]
        return {"intent": intent}
    except Exception as e:
        # Print the exception details for debugging
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

