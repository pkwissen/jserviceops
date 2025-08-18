import pandas as pd

# category_count.py
def category_wise_count(df):
    def normalize_category(text):
        if pd.isna(text):
            return "Unknown"
        return " ".join(text.strip().title().split())

    if 'Category' not in df.columns:
        return pd.DataFrame(columns=["Category", "Ticket Count"])

    df['Category'] = df['Category'].apply(normalize_category)

    category_counts = (
        df['Category']
        .value_counts()
        .reset_index()
    )
    category_counts.columns = ['Category', 'Ticket Count']

    return category_counts
