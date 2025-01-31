from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import logging
from typing import Dict
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReviewRequest(BaseModel):
    review: str 

class SentimentResponse(BaseModel):
    original_text: str
    processed_text: str
    sentiment_prediction: str
    confidence: float

class HealthResponse(BaseModel):
    status: bool
    model_loaded: bool
    vectorizer_loaded: bool

model = None
vectorizer = None

nltk.download('punkt')
nltk.download('stopwords')

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer
    try:
        # Load the model and vectorizer
        with open('model.pkl', 'rb') as file:
            model, vectorizer = pickle.load(file)
        
        # Verify loaded objects
        logger.info(f"Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")
        logger.info(f"Model classes: {model.classes_}")
        logger.info("Model and vectorizer loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model and vectorizer: {str(e)}")
        raise
    
    yield
    
    logger.info("Shutting down and cleaning up resources...")
    model = None
    vectorizer = None

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment of movie reviews",
    version="1.0.0",
    lifespan=lifespan
)

def clean_html(text: str) -> str:
    """Remove HTML tags from text"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def convert_lower(text: str) -> str:
    """Convert text to lowercase"""
    return text.lower()

def remove_special(text: str) -> str:
    """Remove special characters"""
    x = ''
    for i in text:
        if i.isalnum():
            x = x + i
        else:
            x = x + ' '
    return x

def remove_stopwords(text: str) -> str:
    """Remove stop words"""
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def perform_stemming(text: str) -> str:
    """Perform Porter Stemming"""
    words = word_tokenize(text)
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def preprocess_text(text: str) -> str:
    """Apply all preprocessing steps"""
    text = clean_html(text)
    text = convert_lower(text)
    text = remove_special(text)
    text = remove_stopwords(text)
    text = perform_stemming(text)
    return text

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check if the service is healthy"""
    return HealthResponse(
        status=True,
        model_loaded=model is not None,
        vectorizer_loaded=vectorizer is not None
    )

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: ReviewRequest) -> Dict[str, str]:
    """
    Predict sentiment from review text
    """
    try:
        if model is None or vectorizer is None:
            raise HTTPException(
                status_code=500, 
                detail="Model or vectorizer not loaded"
            )
        
        processed_text = preprocess_text(request.review)
        
        text_vectorized = vectorizer.transform([processed_text]).toarray()
        
        proba = model.predict_proba(text_vectorized)[0]
        
        pred_idx = np.argmax(proba)
        prediction = model.classes_[pred_idx]
        
        response = {
            "original_text": request.review,
            "processed_text": processed_text,
            "sentiment_prediction": prediction,
            "confidence": float(proba[pred_idx])  
        }
        
        logger.info(f"Predicted {prediction} with confidence {proba[pred_idx]:.3f}")
        print(response)
        return response
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)