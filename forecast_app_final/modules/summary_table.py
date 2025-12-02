# summary_table_simple.py
import math
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np


def build_hourly_summary(
    full_forecast: pd.DataFrame,
    summary_df: Optional[pd.DataFrame] = None,
    hourly_df: Optional[pd.DataFrame] = None,
    analyst_daily_capacity: int = 10,
    team_size: int = 15,
    working_days_override: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simple hourly summary for full_forecast with columns:
      Date, '1','2',...,'24', Channel

    Returns: (hourly_df_out, meta)
    hourly_df_out columns: Hour(0..23), Tickets, Average/day, Analysts required, Available, Teams available
    """
    meta: Dict[str, Any] = {}

    # Validate input
    if full_forecast is None or not isinstance(full_forecast, pd.DataFrame) or full_forecast.empty:
        rows = [{"Hour": h, "Tickets": 0, "Average/day": np.nan, "Analysts required": 0, "Available": 0, "Teams available": 0} for h in range(24)]
        meta["error"] = "full_forecast missing or empty"
        return pd.DataFrame(rows), meta

    df = full_forecast.copy()
    # make sure column names are strings without extra spaces
    df.columns = [str(c).strip() for c in df.columns]

    # Sum tickets per hour from columns '1'..'24' (map to hours 0..23)
    tickets_by_hour = {h: 0.0 for h in range(24)}
    for i in range(1, 25):
        col = str(i)
        if col in df.columns:
            tickets_by_hour[i - 1] = float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
        else:
            tickets_by_hour[i - 1] = 0.0
    meta["tickets_by_hour"] = tickets_by_hour

    # Compute working days from Date column (exclude Sat & Sun)
    if working_days_override is not None:
        working_days = int(working_days_override)
    else:
        working_days = 0
        if "Date" in df.columns:
            try:
                dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
                if not dates.empty:
                    unique_days = pd.Series(dates.dt.normalize().unique())
                    weekdays = unique_days[unique_days.dt.weekday < 5]
                    working_days = int(len(weekdays)) if len(weekdays) > 0 else int(len(unique_days))
            except Exception:
                working_days = 0
    meta["working_days"] = working_days

    # Determine teams per hour
    # 1) Try hourly_df (expecting hour-like column and team column)
    teams_per_hour = {h: 0 for h in range(24)}
    if hourly_df is not None and isinstance(hourly_df, pd.DataFrame) and not hourly_df.empty:
        hdf = hourly_df.copy()
        cols_low = [str(c).strip().lower() for c in hdf.columns]
        # find team col and hour col
        team_col = None
        hour_col = None
        for c, lc in zip(hdf.columns, cols_low):
            if "team" in lc or "group" in lc:
                team_col = c
            if lc in ("hour", "hr", "slot"):
                hour_col = c
        # fallback: numeric-looking hour column
        if hour_col is None:
            for c in hdf.columns:
                sample = hdf[c].dropna().astype(str).head(20).tolist()
                if sample:
                    ints = sum(1 for v in sample if v.isdigit() and 0 <= int(v) <= 24)
                    if ints >= max(1, len(sample)//2):
                        hour_col = c
                        break
        if team_col and hour_col:
            try:
                hrs = pd.to_numeric(hdf[hour_col], errors="coerce")
                # convert 1..24 to 0..23 if needed
                if hrs.between(1, 24).any():
                    hrs = hrs.apply(lambda x: int(x - 1) if pd.notna(x) and 1 <= int(x) <= 24 else np.nan)
                else:
                    hrs = hrs.apply(lambda x: int(x) if pd.notna(x) and 0 <= int(x) <= 23 else np.nan)
                hdf["_hr"] = hrs
                for hr in range(24):
                    grp = hdf[hdf["_hr"] == hr]
                    teams_per_hour[hr] = int(grp[team_col].nunique()) if not grp.empty else 0
                meta["teams_source"] = "hourly_df"
            except Exception:
                meta["teams_source"] = "hourly_df_error"

    # 2) Fallback to summary_df unique teams across all hours
    if all(v == 0 for v in teams_per_hour.values()):
        if summary_df is not None and isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            team_col = None
            for c in summary_df.columns:
                if "team" in str(c).lower() or "group" in str(c).lower():
                    team_col = c
                    break
            if team_col:
                uniq = int(summary_df[team_col].nunique())
                for hr in range(24):
                    teams_per_hour[hr] = uniq
                meta["teams_source"] = "summary_df"
            else:
                meta["teams_source"] = "none_found"
        else:
            meta["teams_source"] = "none_found"
    meta["teams_per_hour"] = teams_per_hour

    # Build final table
    rows = []
    for hr in range(24):
        tickets = tickets_by_hour.get(hr, 0.0)
        avg_day = float(tickets / working_days) if working_days and working_days > 0 else np.nan
        analysts_req = int(math.ceil(tickets / analyst_daily_capacity)) if analyst_daily_capacity and tickets > 0 else 0
        teams_avail = int(teams_per_hour.get(hr, 0))
        available = teams_avail * team_size
        rows.append({
            "Hour": int(hr),
            "Tickets": int(round(tickets)),
            "Average/day": float(round(avg_day, 2)) if not (isinstance(avg_day, float) and np.isnan(avg_day)) else np.nan,
            "Analysts required": analysts_req,
            "Available": int(available),
            "Teams available": teams_avail,
        })

    hourly_df_out = pd.DataFrame(rows, columns=["Hour", "Tickets", "Average/day", "Analysts required", "Available", "Teams available"])
    return hourly_df_out, meta
