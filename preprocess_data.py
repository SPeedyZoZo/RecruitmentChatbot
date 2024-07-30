import json
from app.utils import preprocess

# Load data
with open('data/data.json', 'r') as f:
    data = json.load(f)

# Preprocess responses
for response in data['responses']:
    response['tokens'] = preprocess(response['text'])

# Save the processed data
with open('data/processed_data.json', 'w') as f:
    json.dump(data, f, indent=4)
