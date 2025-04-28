import streamlit as st
from transformers import pipeline

# Load the sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

st.title("Sentiment Analysis Demo")

# Creating a text input for user to enter a sentence
user_input = st.text_input("Enter a sentence to analyze: ", "I love using Streamlit!")

if user_input:
    try:
        # Performing sentiment analysis
        result = sentiment_pipeline(user_input)[0]
        sentiment = result['label']
        score = result['score']

        # Displaying the results
        st.header("Analysis Result")
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence Score: {score:.2f}")

        # Visualizing confidence score
        st.progress(score)

        # Providing interpretation
        if sentiment == "POSITIVE":
            st.success("The input text has a positive sentiment.")
        else:
            st.error("The input text has a negative sentiment.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
import streamlit as st
from transformers import pipeline
import pandas as pd

# Custom CSS to improve the app's appearance
st.markdown("""

    """, unsafe_allow_html=True)

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", ["distilbert-base-uncased-finetuned-sst-2-english", "roberta-large-mnli"])

# Cache the model loading
@st.cache_resource
def load_model(name):
    return pipeline("sentiment-analysis", model=name)

sentiment_pipeline = load_model(model_name)

st.title("Enhanced Sentiment Analysis Demo")

# Creating two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown('Enter your text:', unsafe_allow_html=True)
    user_input = st.text_area("", "I love using Streamlit!")

with col2:
    if user_input:
        result = sentiment_pipeline(user_input)[0]
        sentiment = result['label']
        score = result['score']

        st.markdown('Analysis Result:', unsafe_allow_html=True)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence Score: {score:.2f}")
        st.progress(score)

        if sentiment == "POSITIVE":
            st.success("Positive sentiment detected!")
        else:
            st.error("Negative sentiment detected!")

# Adding some information about the selected model
with st.sidebar.expander("Model Information"):
    st.write(f"Using model: {model_name}")
    st.write("This model analyzes the sentiment of the input text and classifies it as positive or negative.")

# Allows users to download results
if 'result' in locals():
    df = pd.DataFrame([result])
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name="sentiment_analysis_results.csv",
        mime="text/csv",
    )
