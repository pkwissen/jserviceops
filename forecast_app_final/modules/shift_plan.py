import pandas as pd
import numpy as np

TASK_DURATION = {
    "Chat": 15,
    "Phone": 15,
    "Phone59": 15,
    "Self-service": 30
}
TASKS_PER_DAY = {
    "Chat": 13,
    "Phone": 13,
    "Phone59": 13,
    "Self-service": 15
}
MINUTES_PER_ANALYST = 540  # 9 hours

# Corrected shift timing definitions
SHIFT_DEFINITIONS = {
    "shift1": list(range(6, 16)),              # 6 AM - 3 PM
    "shift2": list(range(8, 18)),              # 8 AM - 5 PM
    "shift3": list(range(11, 21)),             # 11 AM - 8 PM
    "shift4": list(range(12, 22)),             # 12 PM - 9 PM
    "shift5": list(range(14, 24)),             # 2 PM - 11 PM
    "shift6": list(range(17, 24)) + list(range(0, 3)),  # 5 PM - 2 AM
    "shift7": list(range(20, 24)) + list(range(0, 6)),  # 8 PM - 5 AM
    "shift8": list(range(22, 24)) + list(range(0, 8)),  # 10 PM - 7 AM
}

SHIFT_TIME_LABELS = {
    "shift1": "6 am - 3 pm",
    "shift2": "8 am - 5 pm",
    "shift3": "11 am - 8 pm",
    "shift4": "12 pm - 9 pm",
    "shift5": "2 pm - 11 pm",
    "shift6": "5 pm - 2 am",
    "shift7": "8 pm - 5 am",
    "shift8": "10 pm - 7 am",
}

def clean_forecast_df(df, channel_name):
    df = df.iloc[1:]
    df.columns = ['Hour', channel_name, 'Avg_for_7_days']
    df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')
    df = df.dropna(subset=['Hour'])
    df['Hour'] = df['Hour'].astype(int)
    df['Avg_for_7_days'] = pd.to_numeric(df['Avg_for_7_days'], errors='coerce').fillna(0).astype(int)
    return df[['Hour', 'Avg_for_7_days']].rename(columns={'Avg_for_7_days': channel_name})

def load_forecast_data(file_path):
    xls = pd.ExcelFile(file_path)
    chat = clean_forecast_df(pd.read_excel(xls, "Chat"), "Chat")
    phone = clean_forecast_df(pd.read_excel(xls, "Phone"), "Phone")
    phone59 = clean_forecast_df(pd.read_excel(xls, "Phone59"), "Phone59")
    self_service = clean_forecast_df(pd.read_excel(xls, "Self-service"), "Self-service")

    forecast_df = chat.merge(phone, on='Hour') \
                      .merge(phone59, on='Hour') \
                      .merge(self_service, on='Hour')
    return forecast_df

def compute_shift_plan(forecast_df):
    shift_plan = []
    for shift_key, shift_hours in SHIFT_DEFINITIONS.items():
        shift_data = forecast_df[forecast_df["Hour"].isin(shift_hours)]
        plan = {
            "shift": shift_key,
            "Shift Timing": SHIFT_TIME_LABELS[shift_key],
            "start": shift_hours[0],
            "end": shift_hours[-1] + 1  # end is exclusive in time blocks
        }
        total_resource = 0
        for channel in ["Chat", "Phone", "Phone59", "Self-service"]:
            total_tasks = shift_data[channel].sum()
            est_by_tasks = total_tasks / TASKS_PER_DAY[channel]
            est_by_minutes = (total_tasks * TASK_DURATION[channel]) / MINUTES_PER_ANALYST
            required_analysts = round(max(est_by_tasks, est_by_minutes))
            plan[channel] = required_analysts
            total_resource += required_analysts
        plan["total_resource"] = total_resource
        shift_plan.append(plan)
    return pd.DataFrame(shift_plan)

def normalize_hour(hour):
    return hour % 24

def is_hour_in_shift(hour, shift_hours):
    return normalize_hour(hour) in [normalize_hour(h) for h in shift_hours]

def normalize(hour):
    return hour % 24

def generate_hourly_distribution(shift_plan_df):
    shift_df = shift_plan_df.copy()
    shift_df["start"] = shift_df["start"].apply(normalize)
    shift_df["end"] = shift_df["end"].apply(normalize)

    hours = list(range(6, 30))  # covers 6 AM to 7 AM next day
    hour_table = []

    for hour in hours:
        norm_hour = normalize(hour)
        active_shifts = []

        for i, row in shift_df.iterrows():
            s, e = row["start"], row["end"]
            if s <= e:
                if s <= norm_hour < e:
                    active_shifts.append(row["shift"])
            else:  # overnight shift
                if norm_hour >= s or norm_hour < e:
                    active_shifts.append(row["shift"])

        total_required = shift_df[shift_df["shift"].isin(active_shifts)]["total_resource"].max()

        original_shares = {}
        for shift in active_shifts:
            shift_row = shift_df[shift_df["shift"] == shift].iloc[0]
            duration = (shift_row["end"] - shift_row["start"]) % 24
            if duration == 0: duration = 24
            original_shares[shift] = shift_row["total_resource"] / duration

        total_original_hourly = sum(original_shares.values())

        scaling_factor = 1
        if total_original_hourly < total_required and total_original_hourly > 0:
            scaling_factor = total_required / total_original_hourly

        hourly_allocation = {shift: round(original_shares[shift] * scaling_factor) for shift in active_shifts}

        total_allocated = sum(hourly_allocation.values())
        hourly_row = {
            "Hour": normalize(hour),
            "Teams": len(active_shifts),
            **{s: hourly_allocation.get(s, 0) for s in shift_df["shift"]},
            "total resources": total_allocated
        }
        hour_table.append(hourly_row)

    df_result = pd.DataFrame(hour_table)
    df_result = df_result[["Hour", "Teams"] + list(shift_df["shift"]) + ["total resources"]]
    df_result = df_result.rename(columns=SHIFT_TIME_LABELS)
    return df_result

def main_pipeline(
    file_path,
    shift_output_excel=None,
    hourly_output_excel=None
):
    # Load forecast
    forecast_df = load_forecast_data(file_path)

    # Compute shift plan
    shift_plan_df = compute_shift_plan(forecast_df)
    if shift_output_excel:
        shift_plan_df.to_excel(shift_output_excel, index=False)

    # Compute hourly distribution
    hourly_df = generate_hourly_distribution(shift_plan_df)
    if hourly_output_excel:
        hourly_df.to_excel(hourly_output_excel, index=False)

    return shift_plan_df, hourly_df

# Note: All file paths must be passed in from app.py or the caller.
