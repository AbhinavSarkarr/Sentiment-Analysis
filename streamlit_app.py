import streamlit as st
import requests
import json
import plotly.graph_objects as go

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎭",
    layout="wide"
)

def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ]
        },
        title = {'text': "Confidence Score (%)"}
    ))
    return fig

def main():
    st.title("Movie Review Sentiment Analyzer")
    st.write("Enter your movie review below to analyze its sentiment.")

    # Check API health
    try:
        health_response = requests.get("http://localhost:8000/health")
        if health_response.status_code == 200:
            st.sidebar.success("API Status: Online")
        else:
            st.sidebar.error("API Status: Offline")
    except requests.exceptions.ConnectionError:
        st.sidebar.error("API Status: Offline")
        st.error("Cannot connect to the API. Please ensure the backend server is running.")
        return

    # Text input for review
    review_text = st.text_area("Review Text", height=150)

    if st.button("Analyze Sentiment"):
        if not review_text:
            st.warning("Please enter a review to analyze.")
            return

        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json={"review": review_text}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Create two columns for results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Analysis Results")
                    sentiment = result["sentiment_prediction"]
                    confidence = result["confidence"]
                    
                    # Display sentiment with appropriate emoji
                    emoji = "😊" if sentiment == "positive" else "☹️"
                    st.markdown(f"### Sentiment: {sentiment.title()} {emoji}")
                    
                    # Display confidence gauge
                    fig = create_confidence_gauge(confidence)
                    st.plotly_chart(fig)

                with col2:
                    st.subheader("Text Processing")
                    st.markdown("**Original Text:**")
                    st.write(result["original_text"])
                    st.markdown("**Processed Text:**")
                    st.write(result["processed_text"])
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {str(e)}")

if __name__ == "__main__":
    main()