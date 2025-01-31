# IMDB Movie Review Sentiment Analysis

This project implements a sentiment analysis model for movie reviews using the IMDB dataset, featuring a FastAPI backend and Streamlit frontend.

## Project Structure
```
.
├── app.py              # FastAPI application
├── streamlit_app.py    # Streamlit frontend
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

1. Clone the repository:
```bash
git clone <repository-url>
cd sentiment-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the FastAPI backend:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

2. Launch the Streamlit frontend:
```bash
streamlit run streamlit_app.py
```

Access the applications at:
- Frontend: http://localhost:8501
- API Documentation: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Components

### Frontend (Streamlit)
Features:
- Interactive web interface
- Real-time sentiment analysis
- Confidence score visualization
- Text preprocessing display
- API health monitoring

### Backend (FastAPI)
Endpoints:
- POST /predict: Sentiment prediction
- GET /health: System health check

Response formats:
```json
# POST /predict
{
    "original_text": "string",
    "processed_text": "string",
    "sentiment_prediction": "string",
    "confidence": "float"
}

# GET /health
{
    "status": true,
    "model_loaded": true,
    "vectorizer_loaded": true
}
```

## Model Details

### Data Processing
1. Text Preprocessing:
   - HTML tag removal
   - Lowercase conversion
   - Special character removal
   - Stopword removal
   - Porter Stemming

2. Feature Engineering:
   - CountVectorizer (max_features=1400)

### Model Architecture
- Algorithm: MultinomialNB
- Training/Test split: 80/20
- Accuracy: ~87%
- Metrics:
  - Precision: 0.88
  - Recall: 0.86
  - F1-Score: 0.87

## Dependencies

Key packages:
- fastapi
- streamlit
- uvicorn
- scikit-learn
- nltk
- pandas
- numpy
- plotly
- requests

See `requirements.txt` for complete list.

## Future Improvements

1. Model Enhancements:
   - TF-IDF vectorization
   - Deep learning models
   - Cross-validation

2. Features:
   - Batch predictions
   - Confidence thresholds
   - Enhanced visualizations

3. Performance:
   - Model optimization
   - Response caching
   - Load balancing

## License
[Specify your license here]
