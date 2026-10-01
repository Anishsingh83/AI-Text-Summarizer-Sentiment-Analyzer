import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Text Insights Engine", layout="centered")
st.title("📝 AI Text Summarizer & Sentiment Analyzer")
st.write("Powered by Hugging Face Transformers")

# 1. Load the pre-trained models safely using full repo names
@st.cache_resource
def load_models():
    # Modern task definition for the text generation pipeline
    summarizer = pipeline("text-generation", model="facebook/bart-large-cnn")
    
    # ADDED 'bhadresh-savani/' namespace prefix to point directly to the correct repository
    analyzer = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
    return summarizer, analyzer

with st.spinner("Loading AI Models... Please wait (This might take a minute on the first run)."):
    summarizer, analyzer = load_models()

# 2. User Input Area
user_text = st.text_area("Paste a long paragraph, article, or review here:", height=200)

if st.button("Analyze Text") and user_text:
    if len(user_text.split()) < 30:
        st.warning("Please enter a longer text (at least 30 words) for a proper summary.")
    else:
        # Create columns for clean layout side-by-side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 AI Summary")
            with st.spinner("Generating summary..."):
                summary_output = summarizer(user_text, max_new_tokens=130, min_new_tokens=30, do_sample=False)
                clean_summary = summary_output[0]['generated_text']
                st.info(clean_summary)
            
        with col2:
            st.subheader("📊 Emotion & Sentiment")
            with st.spinner("Analyzing sentiment..."):
                sentiment_output = analyzer(user_text)
                label = sentiment_output[0]['label'].upper()
                score = round(sentiment_output[0]['score'] * 100, 2)
                st.metric(label=f"Detected Emotion: {label}", value=f"{score}% Confidence")
