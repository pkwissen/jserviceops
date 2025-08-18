import pandas as pd
import re

def generate_analyst_summary(df):
    # Target column name (lowercase for matching)
    target_col_lower = "analyst who closed the ticket"

    # Map of lowercase → actual column names
    col_map = {c.strip().lower(): c for c in df.columns}
    if target_col_lower not in col_map:
        return pd.DataFrame()  # Column not found

    analyst_col = col_map[target_col_lower]

    # Determine month label source
    if "Month" in df.columns and df["Month"].notna().any():
        df["month_label"] = df["Month"].astype(str)
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["month_label"] = df["Date"].dt.strftime("%b,%y")
    else:
        # Fallback: total counts only
        summary = (
            df[analyst_col]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Analyst who closed the ticket", analyst_col: "Grand Total"})
        )
        return summary

    # Pivot table (ticket counts per analyst per month)
    pivot = pd.pivot_table(
        df,
        index=analyst_col,
        columns="month_label",
        values="Ticket No.",
        aggfunc="count",
        fill_value=0,
    )

    # Sort month columns chronologically
    def _parse_label(lbl):
        try:
            return pd.to_datetime(lbl, format="%b,%y")
        except Exception:
            return pd.Timestamp.max

    ordered_months = sorted(pivot.columns.tolist(), key=_parse_label)
    pivot = pivot[ordered_months]

    # Grand Total column
    pivot["Grand Total"] = pivot.sum(axis=1)

    # Replace 0 with "" for display (except Grand Total)
    display = pivot.copy()
    for col in display.columns:
        if col != "Grand Total":
            display[col] = display[col].replace(0, "")

    # Reset index and unify analyst column name
    display = display.reset_index().rename(columns={analyst_col: "Analyst who closed the ticket"})

    return display
