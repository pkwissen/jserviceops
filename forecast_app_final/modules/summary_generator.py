from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd

def generate_summary_table(
    forecast_data_path: str,
    heatmap_option: Optional[str],
    start_dt,
    end_dt,
    hourly_df: Optional[pd.DataFrame] = None,
    analyst_daily_task_completion: int = 10,
    team_size: int = 15,
    diagnostics: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]] or pd.DataFrame:

    diag: Dict[str, Any] = {
        "missing_sheets": [],
        "sheets": {},
        "num_working_days": 0,
        "tickets_per_hour_sample": {}
    }
    xls = pd.ExcelFile(forecast_data_path)

    # choose channels
    if heatmap_option == "Service Now":
        channels = ["Chat", "Phone", "Self-service"]
    elif heatmap_option == "Service Now - Five9 together":
        channels = ["Chat", "Phone59", "Self-service"]
    else:
        # fallback: choose any known channels present
        present = set(xls.sheet_names)
        channels = [c for c in ["Chat", "Phone", "Phone59", "Self-service"] if c in present]

    # compute working days (exclude Sat/Sun)
    start = pd.to_datetime(start_dt).date()
    end = pd.to_datetime(end_dt).date()
    all_dates = pd.date_range(start=start, end=end).date
    working_dates = [d for d in all_dates if pd.Timestamp(d).weekday() < 5]
    num_working_days = len(working_dates)
    diag["num_working_days"] = num_working_days


    chat = pd.read_excel(xls, sheet_name="Chat", skiprows=1).set_index("Hour")
    phone = pd.read_excel(xls, sheet_name="Phone", skiprows=1).set_index("Hour")
    selfs = pd.read_excel(xls, sheet_name="Self-service", skiprows=1).set_index("Hour")
    total_tickets = chat["Chat"] + phone["Phone"] + selfs["Self-service"]
    df_total = total_tickets.reset_index(name="Tickets")
    
    
    df_total["Average/day"] = round(df_total["Tickets"] / num_working_days) if num_working_days > 0 else 0.0
    df_total["Analysts required"] = round(df_total["Tickets"] / (max(1, analyst_daily_task_completion)))
    df_total["Teams available"] = hourly_df['Teams']
    df_total["Analysts available"] = df_total["Teams available"] * int(team_size)
    
    df_out = df_total.copy()

    if diagnostics:
        return df_out, diag
    return df_out


def save_summary_to_excel(df: pd.DataFrame, out_path: str) -> str:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(str(p), index=False)
    return str(p)
