import os
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from groq import Groq

# =========================
# BERTopic model runner
# =========================
def run_bertopic(texts):
    """Run BERTopic on given texts and return topics + model."""
    vectorizer_model = CountVectorizer(stop_words="english")
    topic_model = BERTopic(vectorizer_model=vectorizer_model, language="english")
    topics, _ = topic_model.fit_transform(texts)
    return topics, topic_model

# =========================
# LLM Refinement (Groq only)
# =========================
def refine_labels_with_llm(unique_labels):
    """
    Use Groq LLM to refine BERTopic topic labels into short, clear names.
    Expects a DataFrame with 'Topic ID' and 'Topic Label'.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it before running.")

    client = Groq(api_key=groq_api_key)

    label_list = "\n".join([f"{row['Topic ID']}: {row['Topic Label']}" for _, row in unique_labels.iterrows()])
    prompt = (
        "You are a text classification expert. Here is a list of topic IDs with their current BERTopic labels.\n"
        "Refine each label into a concise, human-friendly category name.\n"
        "Return the output strictly in the format:\n"
        "Topic ID: Refined Label\n\n"
        f"{label_list}"
    )

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )

    refined_text = response.choices[0].message.content.strip()
    final_labels = {}
    for line in refined_text.split("\n"):
        if ":" in line:
            try:
                topic_id, refined_label = line.split(":", 1)
                final_labels[int(topic_id.strip())] = refined_label.strip()
            except ValueError:
                continue

    return final_labels

# =========================
# Main Topic Label Generator
# =========================
def generate_topic_labels(df):
    """Generate BERTopic topics and LLM-refined labels."""
    if "User Response" not in df.columns:
        raise ValueError("Column 'User Response' not found in DataFrame.")

    # Filter empty or placeholder responses
    df = df[df["User Response"].notna()]
    df = df[~df["User Response"].str.strip().str.lower().isin(["", "no comments from the user"])]

    texts = df["User Response"].tolist()
    topics, topic_model = run_bertopic(texts)

    df["Topic ID"] = topics
    df["Topic Label"] = [topic_model.topic_labels_.get(t, "Unknown") for t in topics]
    df = df[df["Topic ID"] != -1]  # remove outliers

    unique_labels = df[["Topic ID", "Topic Label"]].drop_duplicates()
    final_labels = refine_labels_with_llm(unique_labels)

    df["Topic Label"] = df["Topic ID"].map(final_labels)

    return df
