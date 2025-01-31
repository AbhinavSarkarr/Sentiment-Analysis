from pydantic import BaseModel

class ReviewRequest(BaseModel):
    review_text: str

class SentimentResponse(BaseModel):
    original_text: str
    processed_text: str
    sentiment_prediction: str
