from main import score_response  # Replace 'your_module' with the actual module name

def test_score_response():
    response = "I have experience with Python and Java."
    assert score_response(response) > 0  # Expecting a positive score


from fastapi.testclient import TestClient
from main import app  # Assuming your FastAPI app is defined in main.py

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={"query": "I have experience with Python and Java."})
    assert response.status_code == 200
    assert "intent" in response.json()
    assert "score" in response.json()
