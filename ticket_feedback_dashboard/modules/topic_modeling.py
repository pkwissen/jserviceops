import os
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from groq import Groq
from openai import OpenAI as OpenAIClient
from typing import Optional
import json
from pathlib import Path
from typing import Dict
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
def openai_chat_completion(
    prompt: str,
    system: Optional[str] = None,
    model: str = "openai/gpt-4.1",
    max_tokens: int = 220
) -> str:
    """
    Sends prompt to OpenAI-compatible client. Returns plain text.
    """

    client = OpenAIClient(
        base_url="https://wise-gateway.wisseninfotech.com",
        api_key="sk-84HmOIqSTNnBlPL4K4Z95A"
    )

    # Build messages
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Call chat completion
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
        timeout=30
    )

    raw = resp.choices[0].message.content or ""
    return raw.strip()
_TOPIC_LABEL_CACHE = Path("topic_label_cache.json")

def refine_labels_with_llm(unique_labels_df: pd.DataFrame, cache_path: Path = _TOPIC_LABEL_CACHE) -> Dict[int, str]:
    """
    Refine labels using LLM but cache results on disk.
    unique_labels_df: DataFrame with columns ['Topic ID', 'Topic Label'].
    Returns mapping {topic_id: refined_label}.
    """
    # Load existing cache if present
    cache: Dict[str, str] = {}
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as fh:
                cache = json.load(fh)
        except Exception as e:
            print(f"Warning: failed to read topic label cache: {e}")
            cache = {}

    # Turn cache keys to ints for convenience
    cache_int = {int(k): v for k, v in cache.items()}

    # Build list of topic entries from input
    unique_labels_df = unique_labels_df.drop_duplicates(subset=["Topic ID", "Topic Label"])
    input_map = {int(row["Topic ID"]): row["Topic Label"] for _, row in unique_labels_df.iterrows()}

    # Determine which topic ids need LLM refinement
    missing_ids = [tid for tid in input_map.keys() if tid not in cache_int]

    # If nothing new, return cached mapping (only for requested ids)
    if not missing_ids:
        # return labels for only the topics present in input_map
        return {tid: cache_int.get(tid, input_map.get(tid, "")) for tid in input_map.keys()}

    # Build prompt only for missing topics (keep small to reduce tokens)
    label_list = "\n".join([f"{tid}: {input_map[tid]}" for tid in missing_ids])
    prompt = (
        "You are a text classification expert. Here is a list of topic IDs with their current BERTopic labels.\n"
        "Refine each label into a concise, human-friendly category name (one short phrase each).\n"
        "Return the output strictly in the format:\n"
        "Topic ID: Refined Label\n\n"
        f"{label_list}"
    )

    # Call LLM (your openai wrapper)
    refined_text = openai_chat_completion(prompt=prompt, model="openai/gpt-4.1", max_tokens=500)

    # Parse LLM output and update cache_int
    for line in (refined_text or "").splitlines():
        if ":" in line:
            try:
                topic_id_str, refined_label = line.split(":", 1)
                topic_id = int(topic_id_str.strip())
                refined_label = refined_label.strip()
                if refined_label:
                    cache_int[topic_id] = refined_label
            except Exception:
                continue

    # For any still-missing ids (LLM failed), use the original BERTopic label as fallback
    for tid in missing_ids:
        if tid not in cache_int:
            cache_int[tid] = input_map.get(tid, "")

    # Persist cache (use str keys to be JSON serializable)
    try:
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump({str(k): v for k, v in cache_int.items()}, fh, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: failed to write topic label cache: {e}")

    # Return mapping for only the topics that were in the input
    return {tid: cache_int.get(tid, input_map.get(tid, "")) for tid in input_map.keys()}


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
    # get refined labels (will use cache and call LLM only for new topic ids)
    final_labels_map = refine_labels_with_llm(unique_labels)

    # Map: prefer refined label, else fall back to BERTopic label, else "Unknown"
    df["Topic Label"] = df["Topic ID"].apply(lambda t: final_labels_map.get(int(t), topic_model.topic_labels_.get(int(t), "Unknown")))

    return df
