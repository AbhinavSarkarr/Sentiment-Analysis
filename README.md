# IMDB Movie Review Sentiment Analysis

This project implements a sentiment analysis model for movie reviews using the IMDB dataset. It includes data preprocessing, model training, and a FastAPI service for making predictions.

## Project Structure
```
.
├── app.py              # FastAPI application
├── model.pkl           # Trained model and vectorizer
├── requirements.txt    # Project dependencies
├── test_api.py        # API testing script
└── README.md          # This file
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (optional)

### Installation

1. Clone the repository (or download the project files):
```bash
git clone <repository-url>
cd sentiment-analysis
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Acquisition

The IMDB dataset used in this project contains 50,000 movie reviews labeled as positive or negative. The dataset was loaded using the following steps:

1. Initial data loading and preprocessing
2. Text cleaning (HTML tags removal, lowercase conversion)
3. Special character removal
4. Stopword removal
5. Porter Stemming

## Model Training

### Training Process
1. Text vectorization using CountVectorizer (max_features=1400)
2. Model: MultinomialNB (Multinomial Naive Bayes)
3. Train-test split: 80-20

### Model Performance
- Accuracy on test set: ~87%
- Key metrics:
  - Precision (Positive): 0.88
  - Recall (Positive): 0.86
  - F1-Score: 0.87

## Running the API

1. Start the FastAPI server:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

2. Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Testing the API

#### Using Python requests:
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "review": "This movie was fantastic! The acting was great."
}
response = requests.post(url, json=data)
print(response.json())
```

#### Using curl:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"review": "This movie was fantastic!"}'
```

#### Health Check:
```bash
curl http://localhost:8000/health
```

## API Endpoints

### POST /predict
Makes a sentiment prediction for a given movie review.

Request body:
```json
{
    "review": "string"
}
```

Response:
```json
{
    "original_text": "string",
    "processed_text": "string",
    "sentiment_prediction": "string",
    "confidence": "float"
}
```

### GET /health
Checks the health status of the API and model.

Response:
```json
{
    "status": true,
    "model_loaded": true,
    "vectorizer_loaded": true
}
```

## Dependencies

- fastapi
- uvicorn
- scikit-learn
- nltk
- pandas
- numpy
- requests (for testing)

For a complete list of dependencies with versions, see `requirements.txt`.

## Model Details

The sentiment analysis model uses the following approach:

1. Text Preprocessing:
   - HTML tag removal
   - Lowercase conversion
   - Special character removal
   - Stopword removal
   - Porter Stemming

2. Feature Engineering:
   - CountVectorizer with max_features=1400
   - Vocabulary size: 1400 most frequent terms

3. Model Architecture:
   - Algorithm: MultinomialNB
   - Training data size: 40,000 reviews
   - Test data size: 10,000 reviews

4. Performance Metrics:
   - Accuracy: ~87%
   - Confusion Matrix:
     ```
     [[4321  679]
      [ 621 4379]]
     ```
   - Classification Report:
     ```
     precision    recall  f1-score
     negative     0.87    0.86    0.87
     positive     0.87    0.88    0.87
     ```

## Future Improvements

1. Model Enhancements:
   - Try different vectorization techniques (TF-IDF)
   - Experiment with deep learning models
   - Implement cross-validation

2. API Features:
   - Batch prediction endpoint
   - Confidence threshold parameter
   - Detailed sentiment analysis

3. Performance:
   - Model optimization
   - Response caching
   - Load balancing

## License
[Specify your license here]