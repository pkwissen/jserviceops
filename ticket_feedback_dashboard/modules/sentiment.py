import streamlit as st
from transformers import pipeline

labels = ["Strong Negative", "Mild Negative", "Actionable", "Unclear"]

@st.cache_resource(show_spinner=False)
def load_classifier():
    # Use smaller model for faster inference
    model_name = "typeform/distilbert-base-uncased-mnli"  # change to facebook/bart-large-mnli if needed
    return pipeline("zero-shot-classification", model=model_name)

classifier = load_classifier()

def classify_sentiment(text):
    if not text or not str(text).strip():
        return "Unclear"
    result = classifier(str(text), candidate_labels=labels)
    return result["labels"][0]
