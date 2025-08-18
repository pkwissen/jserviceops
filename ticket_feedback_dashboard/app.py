import streamlit as st
import os
import pandas as pd
import re
import plotly.express as px
import plotly.colors as pc
from dotenv import load_dotenv
from modules.sentiment import classify_sentiment, labels as sentiment_labels
from modules.topic_modeling import generate_topic_labels
from modules.analyst_summary import generate_analyst_summary
from modules.category_count import category_wise_count

load_dotenv()

def build_sentiment_summary(df, recompute=False):
    if recompute or "Sentiment" not in df.columns:
        df["Sentiment"] = df["User Response"].fillna("").astype(str).apply(classify_sentiment)
    else:
        df["Sentiment"] = df["Sentiment"].fillna("Unclear")
    sentiment_counts = (
        df["Sentiment"]
        .value_counts()
        .reindex(sentiment_labels, fill_value=0)
        .rename_axis("Sentiment")
        .reset_index(name="Ticket Count")
    )
    return sentiment_counts

def build_topic_summary(df):
    if "Topic Label" not in df.columns:
        return pd.DataFrame()
    return (
        df["Topic Label"]
        .value_counts()
        .rename_axis("Topic Label")
        .reset_index(name="Ticket Count")
    )

def build_category_summary(df):
    if "Category" in df.columns:
        return category_wise_count(df)
    else:
        return pd.DataFrame()

def display_summary_with_chart(summary_df, full_df, label_column, count_column):
    if summary_df.empty:
        st.warning(f"No data available for {label_column}")
        return
    
    # Calculate percentages
    total_tickets = full_df["Ticket No."].nunique() if "Ticket No." in full_df.columns else 0
    if total_tickets > 0:
        summary_df["Percentage (%)"] = (summary_df[count_column] / total_tickets * 100).round(2)

    # Sort for consistent ordering
    summary_df = summary_df.sort_values(by=count_column, ascending=False)

    # Default: Gradient Red→Blue
    colorscale = px.colors.diverging.RdBu[::-1]
    counts = summary_df[count_column].to_numpy(dtype=float)
    norm_counts = (counts - counts.min()) / (counts.max() - counts.min()) if counts.max() != counts.min() else np.zeros_like(counts)
    default_color_map = {
        label: colorscale[int(norm * (len(colorscale)-1))]
        for label, norm in zip(summary_df[label_column], norm_counts)
    }

    # Special case for Sentiment chart
    if label_column.lower() == "sentiment":
        sentiment_palette = px.colors.qualitative.Set2  # softer set
        color_map = {}
        used_colors = iter(sentiment_palette)
        for label in summary_df[label_column]:
            if label.lower() == "strong negative":
                color_map[label] = "red"
            else:
                color_map[label] = next(used_colors, "#cccccc")
    else:
        color_map = default_color_map

    # Display table
    st.dataframe(summary_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Pie chart
    fig = px.pie(
        summary_df,
        names=label_column,
        values=count_column,
        hole=0.3,
        title=f"{label_column} Distribution",
        color=label_column,
        color_discrete_map=color_map
    )
    fig.update_traces(textinfo="label+percent", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

def display_clickable_summary(summary_df, full_df, key_prefix, label_column, count_column):
    display_summary_with_chart(summary_df, full_df, label_column, count_column)
    options = summary_df[label_column].tolist()
    selected = st.selectbox(f"Select {label_column}", options, key=f"{key_prefix}_select")
    if selected:
        st.markdown(f"### 🔍 Records for {label_column}: `{selected}`")
        filtered_df = full_df[full_df[label_column] == selected]
        st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True, hide_index=True)

def sort_agent_names(df, agent_column="Analyst who closed the ticket"):
    df = df.copy()
    col_map = {c.strip().lower(): c for c in df.columns}
    if agent_column.lower() not in col_map:
        return df
    real_col = col_map[agent_column.lower()]
    df["Agent_num"] = df[real_col].apply(
        lambda x: int(re.search(r'(\d+)$', str(x)).group(1)) if re.search(r'(\d+)$', str(x)) else -1
    )
    df = df.sort_values("Agent_num").drop(columns="Agent_num")
    return df

def main():
    if st.button("Back to Homepage", key="heat_back_homepage"):
        st.session_state["current_app"] = "Homepage"
        st.rerun()  # Navigate back to the homepage

    st.set_page_config(page_title="Ticket Feedback Dashboard", layout="wide")
    st.title("🎫 Ticket Feedback Dashboard (Poor Ratings)")

    project_base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_base_dir, "processed_output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "feedback_with_topics_and_sentiment.xlsx")

    selected_option = st.radio(
        "Choose an option",
        ["-- Select an option --", "Upload and Process New File", "View Last Processed Output"],
        index=0
    )

    if selected_option == "Upload and Process New File":
        uploaded_file = st.file_uploader("Upload Excel File (PoorRatings-Tracker.xlsx)", type=["xlsx"])
        if uploaded_file:
            df = pd.read_excel(uploaded_file, sheet_name="Sheet1")
            df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('\r', ' ').str.title()

            st.subheader("1. Category")
            cat_df = build_category_summary(df)
            display_clickable_summary(cat_df, df, "category", "Category", "Ticket Count")

            st.subheader("2. Sentiments")
            sentiment_counts = build_sentiment_summary(df, recompute=True)
            display_clickable_summary(sentiment_counts, df, "sentiment", "Sentiment", "Ticket Count")

            st.subheader("3. Topic Labels using BERTopic + LLM")
            df = generate_topic_labels(df)
            topic_counts = build_topic_summary(df)
            display_clickable_summary(topic_counts, df, "topic", "Topic Label", "Ticket Count")

            st.subheader("4. Poor rating for closed ticket each month")
            analyst_df = generate_analyst_summary(df)
            analyst_df = sort_agent_names(analyst_df)
            st.dataframe(analyst_df.reset_index(drop=True), hide_index=True)

            df.to_excel(output_path, index=False)

    elif selected_option == "View Last Processed Output":
        if os.path.exists(output_path):
            df = pd.read_excel(output_path)
            st.subheader("1. Category")
            cat_df = build_category_summary(df)
            display_clickable_summary(cat_df, df, "category", "Category", "Ticket Count")

            st.subheader("2. Sentiments")
            sentiment_counts = build_sentiment_summary(df, recompute=False)
            display_clickable_summary(sentiment_counts, df, "sentiment", "Sentiment", "Ticket Count")

            st.subheader("3. Topic Labels")
            topic_counts = build_topic_summary(df)
            display_clickable_summary(topic_counts, df, "topic", "Topic Label", "Ticket Count")

            st.subheader("4. Poor rating for closed ticket each month")
            analyst_df = generate_analyst_summary(df)
            analyst_df = sort_agent_names(analyst_df)
            st.dataframe(analyst_df.reset_index(drop=True), hide_index=True)
        else:
            st.warning("⚠️ No previously processed file found.")
    else:
        st.info("👇 Please select an option to begin.")

if __name__ == "__main__":
    main()
